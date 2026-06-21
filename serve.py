"""
TRELLIS.2 inference server — vLLM-style persistent service.

The model pipeline is loaded once at startup and kept resident in RAM/VRAM.
Clients submit an image and receive a textured GLB (no video) over HTTP or
WebSocket. Generations are serialized through a single-GPU work queue so the
server can accept many concurrent connections while running one job at a time.

Run:
    python serve.py --host 0.0.0.0 --port 8000

Environment variables:
    TRELLIS2_MODEL_PATH   Path/HF repo of the weights (default /models/microsoft/TRELLIS.2-4B)
    TRELLIS2_LOW_VRAM     "1" (default) offloads submodules to CPU between steps;
                          "0" keeps the whole pipeline resident in VRAM (needs more memory).
    TRELLIS2_PIPELINE     Default pipeline type: 512 / 1024 / 1024_cascade / 1536_cascade
"""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
import asyncio
import base64
import io
import tempfile
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel


MODEL_PATH = os.environ.get("TRELLIS2_MODEL_PATH", "/models/microsoft/TRELLIS.2-4B")
LOW_VRAM = os.environ.get("TRELLIS2_LOW_VRAM", "1") not in ("0", "false", "False")
DEFAULT_PIPELINE = os.environ.get("TRELLIS2_PIPELINE", "1024_cascade")


class _State:
    pipeline: Optional[Trellis2ImageTo3DPipeline] = None
    # Only one generation may touch the GPU at a time; this lock turns the
    # server into a single-worker queue while still accepting many connections.
    gpu_lock: asyncio.Lock = asyncio.Lock()
    ready: bool = False
    loaded_at: float = 0.0
    busy: bool = False


state = _State()


# --------------------------------------------------------------------------- #
# Generation parameters
# --------------------------------------------------------------------------- #
class GenParams(BaseModel):
    seed: int = 42
    pipeline_type: Optional[str] = None          # default -> DEFAULT_PIPELINE
    texture_size: int = 4096
    decimation_target: int = 1000000
    simplify: int = 16777216                     # nvdiffrast vertex limit
    max_num_tokens: int = 49152
    preprocess_image: bool = True


def _load_pipeline():
    print(f"[trellis2-serve] Loading pipeline from: {MODEL_PATH}")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_PATH)
    pipeline.low_vram = LOW_VRAM
    pipeline.default_pipeline_type = DEFAULT_PIPELINE
    pipeline.cuda()
    state.pipeline = pipeline
    state.ready = True
    state.loaded_at = time.time()
    mode = "low-VRAM (CPU offload)" if LOW_VRAM else "fully resident in VRAM"
    print(f"[trellis2-serve] Pipeline ready ({mode}); default type={DEFAULT_PIPELINE}")


def _run_generation(image: Image.Image, params: GenParams) -> bytes:
    """Blocking image -> GLB. Must be called inside the GPU lock."""
    pipeline = state.pipeline
    mesh = pipeline.run(
        image,
        seed=params.seed,
        pipeline_type=params.pipeline_type or DEFAULT_PIPELINE,
        max_num_tokens=params.max_num_tokens,
        preprocess_image=params.preprocess_image,
    )[0]
    mesh.simplify(params.simplify)

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=params.decimation_target,
        texture_size=params.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )

    # Export to a temp file then read bytes (to_glb returns a trimesh object).
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        glb.export(tmp_path, extension_webp=False)
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    torch.cuda.empty_cache()
    return data


async def _generate(image: Image.Image, params: GenParams) -> bytes:
    async with state.gpu_lock:
        state.busy = True
        try:
            return await asyncio.to_thread(_run_generation, image, params)
        finally:
            state.busy = False


def _decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load synchronously so /health only reports ready once the model is up.
    await asyncio.to_thread(_load_pipeline)
    yield
    state.pipeline = None
    state.ready = False


app = FastAPI(title="TRELLIS.2 Inference Server", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    if not state.ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "busy": state.busy}


@app.get("/info")
async def info():
    return {
        "model_path": MODEL_PATH,
        "ready": state.ready,
        "busy": state.busy,
        "low_vram": LOW_VRAM,
        "default_pipeline_type": DEFAULT_PIPELINE,
        "loaded_at": state.loaded_at,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/generate")
async def generate(
    image: UploadFile = File(..., description="Input image (png/jpg/webp)"),
    seed: int = Form(42),
    pipeline_type: Optional[str] = Form(None),
    texture_size: int = Form(4096),
    decimation_target: int = Form(1000000),
    simplify: int = Form(16777216),
    max_num_tokens: int = Form(49152),
    preprocess_image: bool = Form(True),
):
    """Multipart upload -> binary GLB response."""
    if not state.ready:
        return JSONResponse({"error": "model still loading"}, status_code=503)
    params = GenParams(
        seed=seed, pipeline_type=pipeline_type, texture_size=texture_size,
        decimation_target=decimation_target, simplify=simplify,
        max_num_tokens=max_num_tokens, preprocess_image=preprocess_image,
    )
    try:
        img = _decode_image(await image.read())
    except Exception as e:
        return JSONResponse({"error": f"invalid image: {e}"}, status_code=400)
    try:
        glb = await _generate(img, params)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    return Response(
        content=glb,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": 'attachment; filename="output.glb"'},
    )


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    """
    Protocol:
      client -> {"image_base64": "...", ...params}
      server -> {"stage": "queued"|"processing"|"exporting"|"done"|"error", ...}
      final  -> {"stage": "done", "glb_base64": "..."}
    """
    await ws.accept()
    try:
        req = await ws.receive_json()
        if not state.ready:
            await ws.send_json({"stage": "error", "message": "model still loading"})
            await ws.close()
            return
        try:
            img = _decode_image(base64.b64decode(req.pop("image_base64")))
        except Exception as e:
            await ws.send_json({"stage": "error", "message": f"invalid image: {e}"})
            await ws.close()
            return
        params = GenParams(**{k: v for k, v in req.items() if k in GenParams.model_fields})

        queued = state.gpu_lock.locked()
        await ws.send_json({"stage": "queued", "queued": queued})

        async with state.gpu_lock:
            state.busy = True
            await ws.send_json({"stage": "processing"})
            t0 = time.time()
            try:
                glb = await asyncio.to_thread(_run_generation, img, params)
            finally:
                state.busy = False
            await ws.send_json({
                "stage": "done",
                "elapsed_sec": round(time.time() - t0, 2),
                "glb_base64": base64.b64encode(glb).decode("ascii"),
            })
        await ws.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await ws.send_json({"stage": "error", "message": str(e)})
            await ws.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 inference server")
    parser.add_argument("--host", default=os.environ.get("TRELLIS2_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TRELLIS2_PORT", "8000")))
    args = parser.parse_args()

    import uvicorn
    # Single worker: the model is loaded once per process and the GPU is shared.
    uvicorn.run(app, host=args.host, port=args.port, workers=1, ws_max_size=64 * 1024 * 1024)


if __name__ == "__main__":
    main()
