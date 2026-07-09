from typing import *
import os
import logging
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
import trimesh
import trimesh.visual
from flex_gemm.ops.grid_sample import grid_sample_3d
import nvdiffrast.torch as dr
import cumesh


logger = logging.getLogger("o_voxel.postprocess")
# serve.py configures logging only on its own "trellis2.serve" logger and never
# touches the root logger, so this library logger would otherwise inherit the
# root default (WARNING) with no handler and drop every INFO line. Make it
# self-sufficient: its own stderr handler at INFO, and no propagation so it
# won't double-log if the root logger later gains handlers.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Optional cap on the dual-contouring grid resolution, decoupled from the
# decoder resolution. The DC dense grid is O(resolution**3), so lowering this
# cuts the peak cubically *and* runs faster; attribute fidelity is preserved
# because colours/PBR are re-projected from the original hi-res mesh via the BVH
# and the remesh is decimated to ``decimation_target`` regardless. 0 = no cap
# (current behaviour). Only the geometric tessellation of the remesh coarsens.
REMESH_MAX_RES = int(os.environ.get("OVOXEL_REMESH_MAX_RES", "0"))

# Per-stage VRAM tracing for to_glb. Set OVOXEL_VRAM_LOG=0 to disable.
VRAM_LOG = os.environ.get("OVOXEL_VRAM_LOG", "1") != "0"

_vram_prev = None


def _log_vram(label: str, detail: str = "") -> None:
    """Log driver-level VRAM at a stage boundary.

    ``mem_get_info`` reports the driver's free/total, so it captures raw
    cudaMalloc allocations from CuMesh / flex_gemm / nvdiffrast that PyTorch's
    own memory stats never see. ``max_memory_reserved`` additionally catches a
    torch-side transient that was already freed by the time we sample here.
    ``detail`` carries optional per-stage context (e.g. vertex/face counts).
    """
    global _vram_prev
    if not (VRAM_LOG and torch.cuda.is_available()):
        return
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1e9
    delta = 0.0 if _vram_prev is None else used - _vram_prev
    _vram_prev = used
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_reserved() / 1e9
    logger.info("[vram] %-26s driver_used=%.2fG (%+.2fG) torch_reserved=%.2fG "
                "torch_peak_reserved=%.2fG%s", label, used, delta, reserved, peak,
                f"  {detail}" if detail else "")
    torch.cuda.reset_peak_memory_stats()


# Creating an nvdiffrast CUDA context allocates GPU resources and compiles
# device programs on first use. The service handles one request at a time
# (GPU work is serialized), so a single context can be reused across requests
# instead of being rebuilt inside every to_glb call.
_RASTERIZE_CONTEXT = None


def _get_rasterize_context() -> "dr.RasterizeCudaContext":
    global _RASTERIZE_CONTEXT
    if _RASTERIZE_CONTEXT is None:
        _RASTERIZE_CONTEXT = dr.RasterizeCudaContext()
    return _RASTERIZE_CONTEXT


def _pad_uv_seams(img: np.ndarray, valid: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Pad valid texels outward across UV-chart boundaries to kill black seams.

    Grows the valid region one ring per iteration, averaging only valid
    neighbours, so colour is preserved (no background bleed) and the work is
    bounded by ``iterations`` instead of the empty area. Unlike
    ``cv2.inpaint`` (TELEA), which is single-threaded and fills the *entire*
    empty atlas, this only touches a few texels of padding around each chart —
    the only region ever reached by bilinear / mip sampling.

    Args:
        img: HxWxC uint8 texture.
        valid: HxW bool mask, True where ``img`` holds real (baked) data.
        iterations: number of 1-texel dilation rings to fill.
    """
    out = img.astype(np.float32) * valid[..., None]
    cnt = valid.astype(np.float32)
    kernel = np.ones((3, 3), np.float32)
    for _ in range(iterations):
        frontier = (cnt < 0.5) & (cv2.dilate(cnt, kernel) > 0.5)
        if not frontier.any():
            break
        nb_sum = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        nb_cnt = cv2.filter2D(cnt, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        avg = nb_sum / np.maximum(nb_cnt, 1.0)[..., None]
        f = frontier[..., None]
        np.copyto(out, avg, where=f)
        np.copyto(cnt, 1.0, where=frontier)
    return out.clip(0, 255).astype(np.uint8)


def _edge_split_by_angle(
    vertices: np.ndarray, faces: np.ndarray, angle_rad: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Split vertices along edges sharper than ``angle_rad``.

    Reproduces Blender's "Shade Smooth by Angle" (auto-smooth) with the
    *Ignore Sharpness* option: which edges are hard is decided purely from the
    geometric face-to-face angle, ignoring any sharp-edge marks. After the
    split, plain smooth vertex-normal averaging yields flat shading across the
    split (hard) edges and smooth shading everywhere else -- the glTF-bakeable
    equivalent of custom split normals, since glTF stores baked per-vertex
    normals rather than an abstract smoothing angle.

    Faces that meet across a *smooth* edge keep sharing a vertex; faces that
    meet across a *sharp* edge (or a boundary / non-manifold edge) get their own
    copy of the shared vertex.

    Args:
        vertices: (V, 3) float array of vertex positions.
        faces: (F, 3) int array of triangle vertex indices.
        angle_rad: edges whose dihedral angle exceeds this are split.

    Returns:
        ``(new_vertices, new_faces)`` with coincident-position vertices
        duplicated at hard edges. Face count and winding are unchanged.
    """
    V = int(vertices.shape[0])
    F = int(faces.shape[0])
    if F == 0:
        return vertices, faces

    # Per-face unit normals.
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-20

    # Each (face, corner) is a node; smooth edges union the two corners they
    # share at each endpoint. Corner id of face f local vertex k is 3*f + k.
    parent = np.arange(3 * F, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Enumerate every face's three edges as undirected (lo, hi) endpoints,
    # tracking which corner sits at the lo end and which at the hi end.
    fi = np.arange(F)
    lo_l, hi_l, clo_l, chi_l, face_l = [], [], [], [], []
    for e in range(3):
        a = faces[:, e]
        b = faces[:, (e + 1) % 3]
        ca = 3 * fi + e
        cb = 3 * fi + (e + 1) % 3
        a_is_lo = a <= b
        lo_l.append(np.where(a_is_lo, a, b))
        hi_l.append(np.where(a_is_lo, b, a))
        clo_l.append(np.where(a_is_lo, ca, cb))
        chi_l.append(np.where(a_is_lo, cb, ca))
        face_l.append(fi)
    lo = np.concatenate(lo_l)
    hi = np.concatenate(hi_l)
    clo = np.concatenate(clo_l)
    chi = np.concatenate(chi_l)
    efaces = np.concatenate(face_l)

    # Group half-edges by undirected edge key; only manifold edges (exactly two
    # incident faces) can be smoothed across. Sorting makes groups contiguous.
    key = lo.astype(np.int64) * V + hi.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    change = np.empty(key_s.shape[0], dtype=bool)
    change[0] = True
    change[1:] = key_s[1:] != key_s[:-1]
    group_start = np.flatnonzero(change)
    group_len = np.diff(np.append(group_start, key_s.shape[0]))
    manifold = group_start[group_len == 2]
    i0 = order[manifold]
    i1 = order[manifold + 1]

    # Smooth where the angle between the two face normals is within threshold.
    cos_thr = float(np.cos(angle_rad))
    dot = np.einsum("ij,ij->i", fn[efaces[i0]], fn[efaces[i1]])
    smooth = dot >= cos_thr
    si0, si1 = i0[smooth], i1[smooth]

    for a, b in zip(clo[si0].tolist(), clo[si1].tolist()):
        union(a, b)
    for a, b in zip(chi[si0].tolist(), chi[si1].tolist()):
        union(a, b)

    # Compact corner roots into new vertex ids.
    roots = np.array([find(c) for c in range(3 * F)], dtype=np.int64)
    uniq, inv = np.unique(roots, return_inverse=True)
    new_faces = inv.reshape(F, 3).astype(faces.dtype, copy=False)
    corner_vertex = faces.reshape(-1)
    new_vertices = vertices[corner_vertex[uniq]]
    return new_vertices, new_faces


def to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb: Union[list, tuple, np.ndarray, torch.Tensor],
    voxel_size: Union[float, list, tuple, np.ndarray, torch.Tensor] = None,
    grid_size: Union[int, list, tuple, np.ndarray, torch.Tensor] = None,
    decimation_target: int = 1000000,
    texture_size: int = 2048,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    smooth_by_angle: bool = False,
    smooth_angle_deg: float = 30.0,
    alpha_mode: str = 'OPAQUE',
    geometry_only: bool = False,
    verbose: bool = False,
    use_tqdm: bool = False,
    progress_callback: Optional[Callable[[int, str], None]] = None,
):
    """
    Convert an extracted mesh to a GLB file.
    Performs cleaning, optional remeshing, UV unwrapping, and texture baking from a volume.
    
    Args:
        vertices: (N, 3) tensor of vertex positions
        faces: (M, 3) tensor of vertex indices
        attr_volume: (L, C) features of a sparse tensor for attribute interpolation
        coords: (L, 3) tensor of coordinates for each voxel
        attr_layout: dictionary of slice objects for each attribute
        aabb: (2, 3) tensor of minimum and maximum coordinates of the volume
        voxel_size: (3,) tensor of size of each voxel
        grid_size: (3,) tensor of number of voxels in each dimension
        decimation_target: target number of vertices for mesh simplification
        texture_size: size of the texture for baking
        remesh: whether to perform remeshing
        remesh_band: size of the remeshing band
        remesh_project: projection factor for remeshing
        mesh_cluster_threshold_cone_half_angle_rad: threshold for cone-based clustering in uv unwrapping
        mesh_cluster_refine_iterations: number of iterations for refining clusters in uv unwrapping
        mesh_cluster_global_iterations: number of global iterations for clustering in uv unwrapping
        mesh_cluster_smooth_strength: strength of smoothing for clustering in uv unwrapping
        smooth_by_angle: if True, split vertices along edges sharper than
            ``smooth_angle_deg`` before UV unwrap so the exported normals match
            Blender's "Shade Smooth by Angle" (ignore sharpness). Off by default.
        smooth_angle_deg: dihedral-angle threshold in degrees for
            ``smooth_by_angle`` (Blender default 30).
        alpha_mode: glTF alpha mode - 'OPAQUE', 'MASK', or 'BLEND'
        geometry_only: if True, skip UV unwrapping and texture baking and return
            a plain (white, UV-less) mesh after remeshing/simplification/cleanup.
            ``attr_volume``, ``coords`` and ``attr_layout`` may be None.
        verbose: whether to print verbose messages
        use_tqdm: whether to use tqdm to display progress bar
        progress_callback: optional ``(percent, stage)`` progress callback
    """
    def report(percent: int, stage: str) -> None:
        if progress_callback is not None:
            progress_callback(percent, stage)

    report(0, "preparing mesh")
    # Attribute tensors are absent in geometry-only mode; anchor device math to
    # the vertices instead of the (possibly None) voxel coords.
    ref_device = coords.device if coords is not None else vertices.device
    # --- Input Normalization (AABB, Voxel Size, Grid Size) ---
    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=ref_device)
    assert isinstance(aabb, torch.Tensor), f"aabb must be a list, tuple, np.ndarray, or torch.Tensor, but got {type(aabb)}"
    assert aabb.dim() == 2, f"aabb must be a 2D tensor, but got {aabb.shape}"
    assert aabb.size(0) == 2, f"aabb must have 2 rows, but got {aabb.size(0)}"
    assert aabb.size(1) == 3, f"aabb must have 3 columns, but got {aabb.size(1)}"

    # Calculate grid dimensions based on AABB and voxel size
    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=ref_device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=ref_device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size
    
    # Assertions for dimensions
    assert isinstance(voxel_size, torch.Tensor)
    assert voxel_size.dim() == 1 and voxel_size.size(0) == 3
    assert isinstance(grid_size, torch.Tensor)
    assert grid_size.dim() == 1 and grid_size.size(0) == 3
    
    if use_tqdm:
        pbar = tqdm(total=6, desc="Extracting GLB")
    if verbose:
        print(f"Original mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

    # Move data to GPU
    vertices = vertices.cuda()
    faces = faces.cuda()
    
    _log_vram("to_glb:entry")

    # Initialize CUDA mesh handler
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    _log_vram("after input.init", f"verts={vertices.shape[0]} faces={faces.shape[0]}")
    
    # --- Initial Mesh Cleaning ---
    # Fills holes as much as we can before processing
    mesh.fill_holes(max_hole_perimeter=3e-2)
    if verbose:
        print(f"After filling holes: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
    vertices, faces = mesh.read()
    report(10, "initial mesh cleanup complete")
    if use_tqdm:
        pbar.update(1)
        
    # Build BVH for the current mesh to guide remeshing and project texture
    # samples back to the original high-resolution surface.
    if use_tqdm:
        pbar.set_description("Building BVH")
    if verbose:
        print(f"Building BVH for current mesh...", end='', flush=True)
    bvh = cumesh.cuBVH(vertices, faces)
    report(20, "BVH built")
    _log_vram("after bvh build")
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
        
    if use_tqdm:
        pbar.set_description("Cleaning mesh")
    if verbose:
        print("Cleaning mesh...")
    
    # --- Branch 1: Standard Pipeline (Simplification & Cleaning) ---
    if not remesh:
        report(25, "simplifying and cleaning mesh")
        # Step 1: Aggressive simplification (3x target)
        mesh.simplify(decimation_target * 3, verbose=verbose)
        if verbose:
            print(f"After inital simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 2: Clean up topology (duplicates, non-manifolds, isolated parts)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After initial cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 3: Final simplification to target count
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After final simplification: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Step 4: Final Cleanup loop
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        if verbose:
            print(f"After final cleanup: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
            
        # Step 5: Unify face orientations
        mesh.unify_face_orientations()
    
    # --- Branch 2: Remeshing Pipeline ---
    else:
        report(25, "remeshing with dual contouring")
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()

        # Cap the DC grid resolution to bound the O(resolution**3) peak. This is
        # the dominant memory consumer of the whole to_glb call.
        if REMESH_MAX_RES and resolution > REMESH_MAX_RES:
            logger.info("capping remesh resolution %d -> %d", resolution, REMESH_MAX_RES)
            resolution = REMESH_MAX_RES

        # Release PyTorch's cached (reserved-but-unused) blocks back to the driver
        # so CuMesh's raw cudaMalloc for the DC grid isn't competing with them.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Perform Dual Contouring remeshing (rebuilds topology).
        rm = cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center = center,
            scale = (resolution + 3 * remesh_band) / resolution * scale,
            resolution = resolution,
            band = remesh_band,
            project_back = remesh_project, # Snaps vertices back to original surface
            verbose = verbose,
            bvh = bvh,
        )
        _log_vram("after remesh_dc", f"verts={rm[0].shape[0]} faces={rm[1].shape[0]}")
        mesh.init(*rm)
        if verbose:
            print(f"After remeshing: {mesh.num_vertices} vertices, {mesh.num_faces} faces")
        
        # Simplify and clean the remeshed result (similar logic to above)
        mesh.simplify(decimation_target, verbose=verbose)
        if verbose:
            print(f"After simplifying: {mesh.num_vertices} vertices, {mesh.num_faces} faces")

    report(50, "mesh topology complete")
    _log_vram("after remesh+simplify")

    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")

    # --- Optional Smooth-by-Angle (Blender "Shade Smooth by Angle", ignore
    # sharpness) ---
    # Split vertices along edges sharper than the threshold so that the smooth
    # vertex-normal averaging done later reproduces auto-smooth shading: flat
    # across creases, smooth elsewhere. Done here on the welded, decimated mesh
    # -- before UV unwrap -- so smoothing groups follow true geometry and the
    # new hard edges naturally fall on UV seams. Benefits both the geometry-only
    # (white) and textured outputs, which read normals from this mesh.
    if smooth_by_angle:
        report(52, "smoothing by angle")
        sv, sf = mesh.read()
        sf_np = sf.detach().cpu().numpy()
        nsv, nsf = _edge_split_by_angle(
            sv.detach().cpu().numpy(),
            sf_np.astype(np.int64),
            np.radians(smooth_angle_deg),
        )
        mesh.init(
            torch.from_numpy(np.ascontiguousarray(nsv, dtype=np.float32)).cuda(),
            torch.from_numpy(np.ascontiguousarray(nsf.astype(sf_np.dtype))).cuda(),
        )
        if verbose:
            print(f"After smooth-by-angle split: {mesh.num_vertices} vertices, "
                  f"{mesh.num_faces} faces")
        _log_vram("after smooth_by_angle")

    # --- Geometry-only (white mesh) short-circuit ---
    # Skip UV unwrapping and texture baking entirely: return a plain mesh with
    # vertex normals but no UVs and no material.
    if geometry_only:
        report(70, "reading geometry")
        out_vertices, out_faces = mesh.read()
        mesh.compute_vertex_normals()
        out_normals = mesh.read_vertex_normals()

        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        normals_np = out_normals.cpu().numpy()

        # Swap Y and Z axes, invert Y (same GLB coordinate convention as the
        # textured path) so mesh-only and full outputs are oriented identically.
        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]

        white_mesh = trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
        )
        if use_tqdm:
            pbar.close()
        report(100, "white mesh ready")
        return white_mesh

    # --- UV Parameterization ---
    if use_tqdm:
        pbar.set_description("Parameterizing new mesh")
    if verbose:
        print("Parameterizing new mesh...")
    
    report(55, "unwrapping UVs")
    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]
    
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    report(70, "UV unwrapping complete")
    _log_vram("after uv_unwrap")
    
    # --- Texture Baking (Attribute Sampling) ---
    if use_tqdm:
        pbar.set_description("Sampling attributes")
    if verbose:
        print("Sampling attributes...", end='', flush=True)
        
    report(75, "sampling texture attributes")
    # Reuse the cached differentiable rasterizer context across requests.
    ctx = _get_rasterize_context()
    # Prepare UV coordinates for rasterization (rendering in UV space)
    uvs_rast = torch.cat([out_uvs * 2 - 1, torch.zeros_like(out_uvs[:, :1]), torch.ones_like(out_uvs[:, :1])], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)
    
    # Rasterize in chunks to save memory
    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i+100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i # Store face ID in alpha channel
        rast = torch.where(mask_chunk, rast_chunk, rast)
    
    # Mask of valid pixels in texture
    mask = rast[0, ..., 3] > 0
    _log_vram("after rasterize")

    # Interpolate 3D positions in UV space (finding 3D coord for every texel)
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]

    # Map these positions back to the *original* high-res mesh to get accurate attributes
    # This corrects geometric errors introduced by simplification/remeshing
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]] # (N_new, 3, 3)
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
    _log_vram("after bvh project")

    # Trilinear sampling from the attribute volume (Color, Material props)
    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )
    _log_vram("after grid_sample_3d")
    if use_tqdm:
        pbar.update(1)
    if verbose:
        print("Done")
    report(90, "texture attributes sampled")
    
    # --- Texture Post-Processing & Material Construction ---
    if use_tqdm:
        pbar.set_description("Finalizing mesh")
    if verbose:
        print("Finalizing mesh...", end='', flush=True)
    
    report(92, "finalizing textures and material")
    mask = mask.cpu().numpy()

    # Scale, clamp, and cast to uint8 on the GPU before the single GPU->CPU
    # copy. This moves the per-texel arithmetic onto the GPU and transfers a
    # quarter of the bytes (uint8 instead of float32).
    attrs_np = (attrs * 255).clamp_(0, 255).to(torch.uint8).cpu().numpy()

    # Pad gaps to prevent black seams at UV boundaries. Pack the scalar PBR
    # channels together so they share a single fill pass.
    base_color = _pad_uv_seams(attrs_np[..., attr_layout['base_color']], mask)
    pbr = _pad_uv_seams(np.concatenate([
        attrs_np[..., attr_layout['metallic']],
        attrs_np[..., attr_layout['roughness']],
        attrs_np[..., attr_layout['alpha']],
    ], axis=-1), mask)
    metallic, roughness, alpha = np.split(pbr, 3, axis=-1)

    # Create PBR material
    # Standard PBR packs Metallic and Roughness into Blue and Green channels
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True if not remesh else False,
    )
    
    # --- Coordinate System Conversion & Final Object ---
    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()
    
    # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1] # Flip UV V-coordinate
    
    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material)
    )
    
    if use_tqdm:
        pbar.update(1)
        pbar.close()
    if verbose:
        print("Done")

    report(100, "GLB mesh ready")
    
    return textured_mesh
