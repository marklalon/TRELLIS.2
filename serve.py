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
    TRELLIS2_PIPELINE     Default pipeline type: 512 / 1024 / 1024_cascade / 1536_cascade
"""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
import asyncio
import base64
import importlib
import io
import logging
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Callable, Optional, TypeVar

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logger = logging.getLogger("trellis2.serve")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

logger.info("Startup progress: importing runtime dependencies")
_runtime_import_started = time.monotonic()
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
logger.info("Startup progress: runtime dependencies imported elapsed=%.2fs",
            time.monotonic() - _runtime_import_started)

# The flow models already use BF16, while decoders and post-processing still
# contain FP32 GEMMs/convolutions. TF32 is substantially faster on recent NVIDIA
# GPUs and retains the FP32 output/range expected by those components.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


MODEL_PATH = os.environ.get("TRELLIS2_MODEL_PATH", "/models/microsoft/TRELLIS.2-4B")
DEFAULT_PIPELINE = os.environ.get("TRELLIS2_PIPELINE", "1024_cascade")
STARTUP_HEARTBEAT_SEC = max(
    1.0, float(os.environ.get("TRELLIS2_STARTUP_HEARTBEAT_SEC", "15"))
)
# Overlap one request's GPU-light post-processing with the next request's
# sampling. Set to 0 to fall back to fully serial generation (e.g. if VRAM is
# too tight to keep one finished mesh resident while another request samples).
OVERLAP_POSTPROCESS = os.environ.get("TRELLIS2_OVERLAP_POSTPROCESS", "1") == "1"
o_voxel = None
T = TypeVar("T")


def _run_startup_stage(label: str, operation: Callable[[], T]) -> T:
    """Run a startup operation with start/end logs and an elapsed heartbeat."""
    started_at = time.monotonic()
    finished = threading.Event()
    logger.info("Startup stage started: %s", label)

    def heartbeat() -> None:
        while not finished.wait(STARTUP_HEARTBEAT_SEC):
            logger.info("Startup stage in progress: %s elapsed=%.2fs",
                        label, time.monotonic() - started_at)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        result = operation()
    except Exception:
        logger.exception("Startup stage failed: %s elapsed=%.2fs",
                         label, time.monotonic() - started_at)
        raise
    finally:
        finished.set()
        heartbeat_thread.join()
    logger.info("Startup stage completed: %s elapsed=%.2fs",
                label, time.monotonic() - started_at)
    return result


class _State:
    pipeline: Optional[object] = None
    # Generation is split into two phases with separate locks so the GPU-light
    # post-processing tail of one request (cumesh UV/BVH/remesh, GLB export —
    # latency-bound work that barely touches the GPU) can overlap the GPU-heavy
    # sampling phase of the next request.
    #
    #   sampling_lock    - held during model sampling + VAE decode (saturates
    #                      the GPU); strictly one at a time.
    #   postprocess_lock - held during to_glb + export; one at a time because it
    #                      guards non-reentrant cumesh state and the shared
    #                      nvdiffrast rasterizer context.
    #
    # A request acquires the two locks in sequence and never holds both at once,
    # so there is no deadlock; request B can sample while request A finishes its
    # post-processing.
    sampling_lock: asyncio.Lock = asyncio.Lock()
    postprocess_lock: asyncio.Lock = asyncio.Lock()
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
    texture_size: int = 2048
    decimation_target: int = 100000
    simplify: int = 16777216                     # nvdiffrast vertex limit
    max_num_tokens: int = 49152
    preprocess_image: bool = True
    texture_sampling_steps: int = 12             # tex SLat sampling steps
    shape_sampling_steps: int = 12               # shape SLat sampling steps


def _load_pipeline():
    global o_voxel

    pipeline_module = _run_startup_stage(
        "importing TRELLIS modules",
        lambda: importlib.import_module("trellis2.pipelines"),
    )
    pipeline_class = pipeline_module.Trellis2ImageTo3DPipeline
    o_voxel = _run_startup_stage(
        "importing mesh post-processing modules",
        lambda: importlib.import_module("o_voxel"),
    )
    # The package contains a compiled extension, so import the installed package
    # first, then overlay its pure-Python postprocessor from the mounted source.
    # Service restarts now pick up post-processing changes without recompiling.
    local_postprocess = os.path.join(
        os.path.dirname(__file__), "o-voxel", "o_voxel", "postprocess.py"
    )
    if os.path.isfile(local_postprocess):
        spec = importlib.util.spec_from_file_location(
            "o_voxel._workspace_postprocess", local_postprocess
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        o_voxel.postprocess = module

    logger.info("Loading pipeline from: %s", MODEL_PATH)
    # The whole pipeline stays resident in VRAM, so load checkpoint weights
    # straight onto the GPU to avoid a separate CPU->GPU copy.
    pipeline = _run_startup_stage(
        "loading model pipeline",
        lambda: pipeline_class.from_pretrained(
            MODEL_PATH,
            progress_callback=lambda message: logger.info(
                "Startup progress: %s", message
            ),
            device="cuda",
        ),
    )
    pipeline.default_pipeline_type = DEFAULT_PIPELINE
    # Moves the remaining submodules (image encoder, background remover) to CUDA;
    # checkpoints loaded with device="cuda" above are already resident so this is
    # near-free for them.
    _run_startup_stage("moving pipeline to CUDA", pipeline.cuda)
    state.pipeline = pipeline
    state.ready = True
    state.loaded_at = time.time()
    logger.info("Pipeline ready (fully resident in VRAM); default type=%s",
                DEFAULT_PIPELINE)

class _ProgressReporter:
    """Per-request progress/timing, shared across the sampling and
    post-processing phases so the reported percentages and elapsed time stay
    continuous even though the two phases run under different locks (and
    possibly different worker threads)."""

    def __init__(self, request_id: str, progress_callback=None):
        self.request_id = request_id
        self.progress_callback = progress_callback
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at

    def report(self, percent: int, stage: str) -> None:
        now = time.monotonic()
        delta = now - self.last_report_at
        self.last_report_at = now
        elapsed = round(now - self.started_at, 2)
        logger.info("[%s] progress=%d%% stage=%s elapsed=%.2fs delta=%.2fs",
                    self.request_id, percent, stage, elapsed, delta)
        if self.progress_callback is not None:
            self.progress_callback(percent, stage, elapsed)


@torch.inference_mode()
def _run_sampling(reporter: "_ProgressReporter", image: Image.Image,
                  params: GenParams):
    """GPU-heavy phase: model sampling + VAE decode. Must run under the
    sampling lock. Returns the decoded mesh (GPU-resident)."""
    pipeline = state.pipeline
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def pipeline_progress(percent: int, stage: str) -> None:
        # The model pipeline is about 70% of the complete request; the rest is
        # mesh cleanup, GLB construction and file export.
        reporter.report(5 + round(percent * 0.65), stage)

    mesh = pipeline.run(
        image,
        seed=params.seed,
        pipeline_type=params.pipeline_type or DEFAULT_PIPELINE,
        max_num_tokens=params.max_num_tokens,
        preprocess_image=params.preprocess_image,
        shape_slat_sampler_params={
            "steps": params.shape_sampling_steps,
        },
        tex_slat_sampler_params={
            "steps": params.texture_sampling_steps,
        },
        progress_callback=pipeline_progress,
    )[0]
    reporter.report(72, "simplifying mesh")
    mesh.simplify(params.simplify)
    return mesh


@torch.inference_mode()
def _run_postprocess(reporter: "_ProgressReporter", mesh, params: GenParams) -> bytes:
    """GPU-light phase: GLB texturing/geometry (cumesh) + export. Must run
    under the post-processing lock. This is the latency-bound tail that barely
    uses the GPU, so it is allowed to overlap the next request's sampling."""
    reporter.report(78, "building GLB textures and materials")

    def postprocess_progress(percent: int, stage: str) -> None:
        reporter.report(78 + round(percent * 0.17), stage)

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
        verbose=False,
        progress_callback=postprocess_progress,
    )

    # Export to a temp file then read bytes (to_glb returns a trimesh object).
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        reporter.report(95, "exporting GLB")
        # WebP texture encoding is markedly faster than PNG's single-threaded
        # zlib path and produces smaller GLBs. Requires Pillow built with WebP
        # support (libwebp-dev) and a client/viewer that understands the
        # EXT_texture_webp glTF extension.
        glb.export(tmp_path, extension_webp=True)
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    reporter.report(100, f"complete ({len(data)} bytes)")
    return data


async def _generate(image: Image.Image, params: GenParams, request_id: str,
                    progress_callback=None) -> bytes:
    """Two-phase image -> GLB.

    Phase 1 (sampling) holds the sampling lock and saturates the GPU. Phase 2
    (post-processing) holds only the post-processing lock, so once phase 1
    releases the sampling lock the next queued request can start sampling while
    this request finishes its GPU-light GLB construction and export.
    """
    reporter = _ProgressReporter(request_id, progress_callback)

    queued = state.sampling_lock.locked()
    logger.info("[%s] queued=%s", request_id, queued)
    wait_started = time.monotonic()
    async with state.sampling_lock:
        state.busy = True
        logger.info("[%s] sampling lock acquired after %.2fs", request_id,
                    time.monotonic() - wait_started)
        try:
            mesh = await asyncio.to_thread(_run_sampling, reporter, image, params)
        finally:
            state.busy = False
        if not OVERLAP_POSTPROCESS:
            # Serial fallback: keep the sampling lock held through the whole
            # post-processing tail so only one request uses the GPU at a time.
            async with state.postprocess_lock:
                return await asyncio.to_thread(_run_postprocess, reporter, mesh, params)

    # Overlap path: the sampling lock is released here; the next request can
    # sample while this one finishes post-processing under its own lock.
    async with state.postprocess_lock:
        return await asyncio.to_thread(_run_postprocess, reporter, mesh, params)


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
        "default_pipeline_type": DEFAULT_PIPELINE,
        "loaded_at": state.loaded_at,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/generate")
async def generate(
    image: UploadFile = File(..., description="Input image (png/jpg/webp)"),
    seed: int = Form(42),
    pipeline_type: Optional[str] = Form(None),
    texture_size: int = Form(2048),
    decimation_target: int = Form(100000),
    simplify: int = Form(16777216),
    max_num_tokens: int = Form(49152),
    preprocess_image: bool = Form(True),
    texture_sampling_steps: int = Form(12),
    shape_sampling_steps: int = Form(12),
):
    """Multipart upload -> binary GLB response."""
    request_id = uuid.uuid4().hex[:8]
    received_at = time.monotonic()
    logger.info("[%s] HTTP request received filename=%r content_type=%r",
                request_id, image.filename, image.content_type)
    if not state.ready:
        logger.warning("[%s] rejected: model still loading", request_id)
        return JSONResponse({"error": "model still loading"}, status_code=503)
    params = GenParams(
        seed=seed, pipeline_type=pipeline_type, texture_size=texture_size,
        decimation_target=decimation_target, simplify=simplify,
        max_num_tokens=max_num_tokens, preprocess_image=preprocess_image,
        texture_sampling_steps=texture_sampling_steps,
        shape_sampling_steps=shape_sampling_steps,
    )
    try:
        image_data = await image.read()
        img = _decode_image(image_data)
        logger.info("[%s] image decoded bytes=%d size=%sx%s mode=%s params=%s",
                    request_id, len(image_data), img.width, img.height, img.mode,
                    params.model_dump())
    except Exception as e:
        logger.warning("[%s] invalid image: %s", request_id, e)
        return JSONResponse({"error": f"invalid image: {e}"}, status_code=400)
    try:
        glb = await _generate(img, params, request_id)
    except Exception as e:
        logger.exception("[%s] generation failed", request_id)
        return JSONResponse({"error": str(e)}, status_code=500)
    logger.info("[%s] HTTP request completed status=200 bytes=%d elapsed=%.2fs",
                request_id, len(glb), time.monotonic() - received_at)
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
    request_id = uuid.uuid4().hex[:8]
    await ws.accept()
    logger.info("[%s] WebSocket connected client=%s", request_id, ws.client)
    try:
        req = await ws.receive_json()
        logger.info("[%s] WebSocket generation request received", request_id)
        if not state.ready:
            logger.warning("[%s] rejected: model still loading", request_id)
            await ws.send_json({"stage": "error", "message": "model still loading"})
            await ws.close()
            return
        try:
            image_data = base64.b64decode(req.pop("image_base64"), validate=True)
            img = _decode_image(image_data)
        except Exception as e:
            logger.warning("[%s] invalid image: %s", request_id, e)
            await ws.send_json({"stage": "error", "message": f"invalid image: {e}"})
            await ws.close()
            return
        params = GenParams(**{k: v for k, v in req.items() if k in GenParams.model_fields})
        logger.info("[%s] image decoded bytes=%d size=%sx%s mode=%s params=%s",
                    request_id, len(image_data), img.width, img.height, img.mode,
                    params.model_dump())

        loop = asyncio.get_running_loop()

        def send_progress(percent, stage, elapsed):
            message = {
                "stage": "processing", "step": stage,
                "progress": percent, "elapsed_sec": elapsed,
                "request_id": request_id,
            }
            future = asyncio.run_coroutine_threadsafe(ws.send_json(message), loop)
            try:
                future.result(timeout=5)
            except Exception:
                # Client progress delivery must never abort GPU work.
                future.cancel()

        queued = state.sampling_lock.locked()
        logger.info("[%s] queued=%s", request_id, queued)
        await ws.send_json({
            "stage": "queued", "queued": queued, "request_id": request_id
        })
        await ws.send_json({"stage": "processing", "progress": 0, "request_id": request_id})

        t0 = time.time()
        # Locking (and the sampling/post-processing phase split) lives in
        # _generate so HTTP and WebSocket share one scheduling path.
        glb = await _generate(img, params, request_id, send_progress)
        await ws.send_json({
            "stage": "done",
            "elapsed_sec": round(time.time() - t0, 2),
            "progress": 100,
            "request_id": request_id,
            "glb_base64": base64.b64encode(glb).decode("ascii"),
        })
        logger.info("[%s] WebSocket request completed bytes=%d elapsed=%.2fs",
                    request_id, len(glb), time.time() - t0)
        await ws.close()
    except WebSocketDisconnect:
        logger.warning("[%s] WebSocket disconnected", request_id)
    except Exception as e:
        logger.exception("[%s] WebSocket generation failed", request_id)
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

    class HealthCheckFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return '"GET /health' not in msg and '"GET /health ' not in msg

    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["formatters"] = {
        name: {**formatter, "fmt": f"%(asctime)s.%(msecs)03d {formatter['fmt']}",
               "datefmt": LOG_DATE_FORMAT}
        for name, formatter in uvicorn.config.LOGGING_CONFIG["formatters"].items()
    }
    # Single worker: the model is loaded once per process and the GPU is shared.
    uvicorn.run(app, host=args.host, port=args.port, workers=1,
                ws_max_size=64 * 1024 * 1024, log_config=log_config)


if __name__ == "__main__":
    main()
