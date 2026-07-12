from typing import *
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import trimesh
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
import o_voxel
import cumesh
import nvdiffrast.torch as dr
import cv2
import flex_gemm


class Trellis2TexturingPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
    """
    model_names_to_load = [
        'shape_slat_encoder',
        'tex_slat_decoder',
        'tex_slat_flow_model_1024'
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        tex_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.tex_slat_sampler = tex_slat_sampler
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Trellis2TexturingPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file)
        args = pipeline._pretrained_args

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])

        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'
        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        super().to(device)
        self.image_cond_model.to(device)
        if self.rembg_model is not None:
            self.rembg_model.to(device)

    def preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Preprocess the input mesh.
        """
        vertices = mesh.vertices
        vertices_min = vertices.min(axis=0)
        vertices_max = vertices.max(axis=0)
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale
        tmp = vertices[:, 1].copy()
        vertices[:, 1] = -vertices[:, 2]
        vertices[:, 2] = tmp
        assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            output = self.rembg_model(input)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        cond = self.image_cond_model(image)
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }
    
    def encode_shape_slat(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = 1024,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> SparseTensor:
        """
        Encode the meshes to structured latent.

        Args:
            mesh (trimesh.Trimesh): The mesh to encode.
            resolution (int): The resolution of mesh
            progress_callback (callable): Optional ``(percent, stage)`` callback on a
                0..100 scale so the caller can stream sub-progress through the two
                heavy steps (dual-grid conversion, then the encoder forward) instead
                of stalling on a single "encoding mesh" report.

        Returns:
            SparseTensor: The encoded structured latent.
        """
        def report(percent: int, stage: str) -> None:
            if progress_callback is not None:
                progress_callback(percent, stage)

        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()

        report(0, "converting mesh to dual grid")
        voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices.cpu(), faces.cpu(),
            grid_size=resolution,
            aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=False,
        )

        report(70, "encoding shape latent")
        vertices = SparseTensor(
            feats=dual_vertices * resolution - voxel_indices,
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
        ).to(self.device)
        intersected = vertices.replace(intersected).to(self.device)

        shape_slat = self.models['shape_slat_encoder'](vertices, intersected)
        return shape_slat

    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
            step_callback (callable): Optional ``(completed_steps, total_steps)``
                callback fired after each sampling step, for progress reporting.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=False,
            tqdm_desc="Sampling texture SLat",
            step_callback=step_callback,
        ).samples

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        ret = self.models['tex_slat_decoder'](slat) * 0.5 + 0.5
        return ret
    
    def postprocess_mesh(
        self,
        mesh: trimesh.Trimesh,
        pbr_voxel: SparseTensor,
        resolution: int = 1024,
        texture_size: int = 1024,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> trimesh.Trimesh:
        def report(percent: int, stage: str) -> None:
            if progress_callback is not None:
                progress_callback(percent, stage)

        vertices = mesh.vertices
        faces = mesh.faces
        normals = mesh.vertex_normals
        vertices_torch = torch.from_numpy(vertices).float().cuda()
        faces_torch = torch.from_numpy(faces).int().cuda()
        report(0, "unwrapping UVs")
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
            uvs = mesh.visual.uv.copy()
            uvs[:, 1] = 1 - uvs[:, 1]
            uvs_torch = torch.from_numpy(uvs).float().cuda()
        else:
            _cumesh = cumesh.CuMesh()
            _cumesh.init(vertices_torch, faces_torch)
            # Same fast UV unwrap path as o_voxel.postprocess.to_glb: use xatlas
            # per-face materials to batch many clusters into a few AddMesh calls,
            # avoiding the O(n_clusters) overhead of stock uv_unwrap.
            _cc_kwargs = {
                "threshold_cone_half_angle_rad": np.radians(90.0),
                "refine_iterations": 0,
                "global_iterations": 1,
                "smooth_strength": 1,
            }
            _uv_verbose = o_voxel.postprocess.UV_PROFILE
            if o_voxel.postprocess._xatlas_supports_materials() and not o_voxel.postprocess.UV_LEGACY:
                out_vertices, out_faces, out_uvs, out_vmaps = o_voxel.postprocess._uv_unwrap_fast(
                    _cumesh, _cc_kwargs, verbose=_uv_verbose)
            else:
                out_vertices, out_faces, out_uvs, out_vmaps = _cumesh.uv_unwrap(
                    compute_charts_kwargs=_cc_kwargs, return_vmaps=True, verbose=_uv_verbose)
            vertices_torch = out_vertices.cuda()
            faces_torch = out_faces.cuda()
            uvs_torch = out_uvs.cuda()
            vmap = out_vmaps.cuda()
            vertices = vertices_torch.cpu().numpy()
            faces = faces_torch.cpu().numpy()
            uvs = uvs_torch.cpu().numpy()
            # Recompute vertex normals on the unwrapped cumesh topology and remap
            # them through vmap (same approach as o_voxel.postprocess.to_glb). The
            # input trimesh's own vertex_normals array does NOT line up with
            # uv_unwrap's vertex ordering, so indexing it by vmap scrambled the
            # normals (they ended up uncorrelated with the geometry, breaking PBR
            # shading in viewers).
            _cumesh.compute_vertex_normals()
            cu_normals = _cumesh.read_vertex_normals()
            normals = cu_normals[vmap.to(cu_normals.device)].cpu().numpy()
                
        # rasterize
        report(30, "rasterizing texture")
        ctx = dr.RasterizeCudaContext()
        uvs_torch = torch.cat([uvs_torch * 2 - 1, torch.zeros_like(uvs_torch[:, :1]), torch.ones_like(uvs_torch[:, :1])], dim=-1).unsqueeze(0)
        rast, _ = dr.rasterize(
            ctx, uvs_torch, faces_torch,
            resolution=[texture_size, texture_size],
        )
        mask = rast[0, ..., 3] > 0
        pos = dr.interpolate(vertices_torch.unsqueeze(0), rast, faces_torch)[0][0]
        
        report(50, "sampling texture voxels")
        attrs = torch.zeros(texture_size, texture_size, pbr_voxel.shape[1], device=self.device)
        attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
            pbr_voxel.feats,
            pbr_voxel.coords,
            shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
            grid=((pos[mask] + 0.5) * resolution).reshape(1, -1, 3),
            mode='trilinear',
        )
        
        # construct mesh
        mask = mask.cpu().numpy()
        base_color = np.clip(attrs[..., self.pbr_attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., self.pbr_attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., self.pbr_attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., self.pbr_attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        # extend
        report(70, "inpainting texture")
        mask = (~mask).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask, 1, cv2.INPAINT_TELEA)[..., None]
        
        report(95, "assembling textured mesh")
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            doubleSided=True,
        )

        # Swap Y and Z axes, invert Y (common conversion for GLB compatibility)
        vertices[:, 1], vertices[:, 2] = vertices[:, 2], -vertices[:, 1]
        normals[:, 1], normals[:, 2] = normals[:, 2], -normals[:, 1]
        uvs[:, 1] = 1 - uvs[:, 1] # Flip UV V-coordinate
        
        textured_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=normals,
            process=False,
            visual=trimesh.visual.TextureVisuals(uv=uvs, material=material)
        )
        
        return textured_mesh
        
    
    @torch.no_grad()
    def run(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        resolution: int = 1024,
        texture_size: int = 2048,
        on_shape_encoded: Optional[Callable[[], None]] = None,
        before_decode: Optional[Callable[[], None]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> trimesh.Trimesh:
        """
        Run the pipeline.

        Args:
            mesh (trimesh.Trimesh): The mesh to texture.
            image (Image.Image): The image prompt.
            seed (int): The random seed.
            tex_slat_sampler_params (dict): Additional parameters for the texture latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
            on_shape_encoded (callable): Optional hook fired after ``encode_shape_slat``
                and before texture sampling. The caller uses it to page the texture
                flow model onto the GPU only once shape encoding (the VRAM-peak stage)
                is done, so the flow weights don't inflate the encoder's activation peak.
            before_decode (callable): Optional hook fired after texture sampling and
                before ``decode_tex_slat``. The caller uses it to page the texture flow
                model back to CPU so its weights don't inflate the decode peak (the
                binding peak for texture-only), mirroring the image-to-3D decode path.
            progress_callback (callable): Optional ``(percent, stage)`` callback on a
                0..100 scale, reported at each stage boundary and per sampling step so
                the caller can stream progress instead of stalling through the ~12s of
                shape encoding, texture sampling and decode.
        """
        def report(percent: int, stage: str) -> None:
            if progress_callback is not None:
                progress_callback(percent, stage)

        def report_encode(percent: int, stage: str) -> None:
            # Shape encoding occupies the 15..25 band of the run.
            report(15 + round(percent * 0.10), stage)

        def report_sampling_step(completed: int, total: int) -> None:
            # Texture sampling occupies the 25..75 band of the run.
            report(25 + round(completed / max(total, 1) * 50), "sampling texture")

        def report_postprocess(percent: int, stage: str) -> None:
            # postprocess_mesh (UV unwrap, rasterize, grid-sample, inpaint) is the
            # long pole of texture-only, so give it a wide 82..98 band with internal
            # steps instead of freezing the client on a single "decoding" report.
            report(82 + round(percent * 0.16), stage)

        report(0, "starting texture pipeline")
        if preprocess_image:
            report(3, "preprocessing image")
            image = self.preprocess_image(image)
        mesh = self.preprocess_mesh(mesh)
        torch.manual_seed(seed)
        # Texture sampling always runs on tex_slat_flow_model_1024 (the dedicated
        # 512 texture model was dropped), so condition at 1024 -- its training
        # distribution -- even when the shape is encoded at 512. RoPE PE handles
        # the 512 shape's smaller coordinate range. Mirrors the image-to-3D 512
        # path (see Trellis2ImageTo3DPipeline.run).
        report(8, "encoding image conditions")
        cond = self.get_cond([image], 1024)
        # The tex_slat_decoder is only needed for decode_tex_slat, well past the
        # shape-encoding VRAM peak. Under CPU offload, page it out so its weights
        # don't inflate that peak; it is restored just before decode below.
        decoder = self.models.get('tex_slat_decoder')
        evict_decoder = getattr(self, '_offload', False) and isinstance(decoder, nn.Module)
        if evict_decoder:
            decoder.cpu()
            torch.cuda.empty_cache()
        try:
            shape_slat = self.encode_shape_slat(
                mesh, resolution, progress_callback=report_encode
            )
            # Return the encoder's transient activations to the driver before the
            # flow model pages in and sampling/decode allocate. Only shape_slat
            # survives encoding, so the freed blocks are pure reserved-pool slack;
            # releasing them here keeps the later stages from stacking on top of
            # (and fragmenting against) the encoder's high-water. Mirrors the
            # decode-boundary empty_cache the image-to-3D path uses.
            if evict_decoder:
                torch.cuda.empty_cache()
            # Shape encoding is the peak-VRAM stage; page the texture flow model in
            # only now (see on_shape_encoded above) so its weights don't stack on
            # that peak.
            if on_shape_encoded is not None:
                on_shape_encoded()
            report(25, "sampling texture")
            tex_model = self.models['tex_slat_flow_model_1024']
            tex_slat = self.sample_tex_slat(
                cond, tex_model,
                shape_slat, tex_slat_sampler_params,
                step_callback=report_sampling_step,
            )
            # Evict the flow model (idle now) before decode, then release its freed
            # blocks, so decode -- the binding peak -- runs without the flow weights
            # or the sampling scratch resident. Order matters: free the flow DiT
            # first, then bring the decoder back into the reclaimed space.
            report(75, "evicting flow models")
            if before_decode is not None:
                before_decode()
            if evict_decoder:
                torch.cuda.empty_cache()
                decoder.to(self._device)
                evict_decoder = False  # restored; skip the finally fallback
            report(78, "decoding texture")
            pbr_voxel = self.decode_tex_slat(tex_slat)
            out_mesh = self.postprocess_mesh(
                mesh, pbr_voxel, resolution, texture_size,
                progress_callback=report_postprocess,
            )
            report(100, "texture sampled")
            return out_mesh
        finally:
            # The image-to-3D pipeline assumes tex_slat_decoder is always resident
            # (it is not offloadable there), so never leave it stranded on the CPU
            # if we bail out early (e.g. a cancelled sampling step).
            if evict_decoder:
                decoder.to(self._device)
