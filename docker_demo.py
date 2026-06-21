"""
Docker demo: Image-to-3D generation using locally mounted model weights.
Model path: /models/microsoft/TRELLIS.2-4B
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# Model path — use env var or default to mounted volume
MODEL_PATH = os.environ.get("TRELLIS2_MODEL_PATH", "/models/microsoft/TRELLIS.2-4B")
INPUT_IMAGE = os.environ.get("TRELLIS2_INPUT_IMAGE", "assets/example_image/T.png")
OUTPUT_VIDEO = os.environ.get("TRELLIS2_OUTPUT_VIDEO", "output/sample.mp4")
OUTPUT_GLB = os.environ.get("TRELLIS2_OUTPUT_GLB", "output/sample.glb")


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 Image-to-3D Demo")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Path to model weights")
    parser.add_argument("--input", default=INPUT_IMAGE, help="Input image path")
    parser.add_argument("--output-video", default=OUTPUT_VIDEO, help="Output video path")
    parser.add_argument("--output-glb", default=OUTPUT_GLB, help="Output GLB path")
    parser.add_argument("--no-render", action="store_true", help="Skip video rendering")
    parser.add_argument("--no-export", action="store_true", help="Skip GLB export")
    args = parser.parse_args()

    print(f"Using model path: {args.model_path}")
    print(f"Input image: {args.input}")

    # 1. Setup Environment Map
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(
            cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB
        ),
        dtype=torch.float32, device='cuda'
    ))

    # 2. Load Pipeline (from local path)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.cuda()
    print("Pipeline loaded successfully.")

    # 3. Load Image & Run
    image = Image.open(args.input)
    print("Running inference...")
    mesh = pipeline.run(image)[0]
    mesh.simplify(16777216)  # nvdiffrast limit

    # 4. Render Video
    if not args.no_render:
        os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
        print("Rendering video...")
        video = render_utils.make_pbr_vis_frames(
            render_utils.render_video(mesh, envmap=envmap)
        )
        imageio.mimsave(args.output_video, video, fps=15)
        print(f"Video saved to: {args.output_video}")

    # 5. Export to GLB
    if not args.no_export:
        os.makedirs(os.path.dirname(args.output_glb), exist_ok=True)
        print("Exporting GLB...")
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1000000,
            texture_size=4096,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True
        )
        glb.export(args.output_glb, extension_webp=False)
        print(f"GLB saved to: {args.output_glb}")

    print("Done!")


if __name__ == "__main__":
    main()
