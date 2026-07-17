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
    TRELLIS2_REMBG_MODEL_PATH  Background-removal model path/repo
    TRELLIS2_PIPELINE     Default pipeline type: 512 / 1024 / 1024_cascade / 1536_cascade
"""
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import argparse
import asyncio
import importlib
import io
import json
import logging
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Callable, Literal, Optional, TypeVar

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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
logger.info("Startup progress: runtime dependencies imported elapsed=%.2fs",
            time.monotonic() - _runtime_import_started)

# The flow models already use BF16, while decoders and post-processing still
# contain FP32 GEMMs/convolutions. TF32 is substantially faster on recent NVIDIA
# GPUs and retains the FP32 output/range expected by those components.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


MODEL_PATH = os.environ.get("TRELLIS2_MODEL_PATH", "/models/microsoft/TRELLIS.2-4B")
REMBG_MODEL_PATH = os.environ.get("TRELLIS2_REMBG_MODEL_PATH")
DEFAULT_PIPELINE = os.environ.get("TRELLIS2_PIPELINE", "512")
STARTUP_HEARTBEAT_SEC = max(
    1.0, float(os.environ.get("TRELLIS2_STARTUP_HEARTBEAT_SEC", "15"))
)
# Timeout for each individual `await ws.receive_*()` call (idle timeout).
# If the client pauses between sending JSON params, image, and mesh (texture
# mode), this timer resets after each successfully received message.
RECV_IDLE_TIMEOUT = float(os.environ.get("TRELLIS2_RECV_IDLE_TIMEOUT", "10"))
# Hard ceiling on the total time a WebSocket connection may spend in the
# input-receiving phase (before generation starts). Measured from
# ``ws.accept()``. A client that connects but never sends data hits this.
WS_TOTAL_TIMEOUT = float(os.environ.get("TRELLIS2_WS_TOTAL_TIMEOUT", "120"))
# Overlap one request's GPU-light post-processing with the next request's
# sampling. Set to 0 to fall back to fully serial generation (e.g. if VRAM is
# too tight to keep one finished mesh resident while another request samples).
OVERLAP_POSTPROCESS = os.environ.get("TRELLIS2_OVERLAP_POSTPROCESS", "1") == "1"
SYNTHETIC_WARMUP = os.environ.get("TRELLIS2_SYNTHETIC_WARMUP", "0") == "1"
SYNTHETIC_WARMUP_DELAY_SEC = 0.0
SYNTHETIC_WARMUP_PIPELINE = DEFAULT_PIPELINE
SYNTHETIC_WARMUP_MODE = "full"
SYNTHETIC_WARMUP_SEED = 1
SYNTHETIC_WARMUP_SHAPE_STEPS = 1
SYNTHETIC_WARMUP_TEXTURE_STEPS = 1
SYNTHETIC_WARMUP_TEXTURE_SIZE = 512
SYNTHETIC_WARMUP_DECIMATION_TARGET = 20000
SYNTHETIC_WARMUP_SIMPLIFY = 500000
# After the pipeline is resident in VRAM, reclaim the transient host memory the
# load leaves behind. Two independent knobs:
#   TRELLIS2_TRIM_MEMORY (default on): gc + empty_cache + glibc malloc_trim, to
#     return the freed CPU allocator arenas (transient per-model CPU construction
#     from load_state_dict(assign=True)) back to the OS. Safe and cheap.
#   TRELLIS2_DROP_PAGE_CACHE (default on): drop the page cache filled by reading
#     the ~20GB of weight files. This is the big reclaim on WSL2, where the page
#     cache otherwise stays pinned inside the VM. Needs root (the container is)
#     and only touches clean cache, but forces a re-read on first cold access, so
#     set it to 0 if you would rather keep the weights hot in cache.
TRIM_MEMORY = os.environ.get("TRELLIS2_TRIM_MEMORY", "1") == "1"
DROP_PAGE_CACHE = os.environ.get("TRELLIS2_DROP_PAGE_CACHE", "1") == "1"
# VRAM guardrails against a single heavy request OOM-ing the process.
#   TRELLIS2_MAX_ACTIVE_TOKENS (default off): reject a request whose sparse
#     structure decodes to more active voxels than this. Peak VRAM scales
#     ~linearly with the token count and it is known ~0.15s in, so this is a
#     cheap predictive cap that fails the request cleanly before the expensive
#     sampling/decode stages. 0 or unset disables it. Tune per GPU: on the log
#     that motivated this (real torch peak ~24G at ~2.4M-vert output), pick a
#     ceiling with headroom below the card's usable VRAM.
#   TRELLIS2_MEM_FRACTION (default 0.0): hard-cap the torch allocator to this
#     fraction of the device's total VRAM. An over-budget allocation then raises
#     a catchable CUDA OOM (failing one request) instead of spilling into slow
#     shared system memory or tripping the OS OOM-killer. Set to 0 to disable.
_max_active_tokens_raw = int(os.environ.get("TRELLIS2_MAX_ACTIVE_TOKENS", "0"))
MAX_ACTIVE_TOKENS = _max_active_tokens_raw if _max_active_tokens_raw > 0 else None
MEM_FRACTION = float(os.environ.get("TRELLIS2_MEM_FRACTION", "0.0"))
o_voxel = None
T = TypeVar("T")


def _trim_host_memory() -> None:
    """Return transient host RAM used during model loading back to the OS.

    The weights are resident in VRAM by this point; the host-side leftovers are
    (1) freed CPU tensors from the per-model construction that glibc keeps in its
    arenas, and (2) the page cache from reading the weight files. Neither is
    needed for inference, but on WSL2 both keep the VM's memory footprint high
    because the VM does not proactively hand reclaimable memory back to Windows.
    """
    import ctypes
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # glibc retains freed heap in per-arena free lists by default; malloc_trim
    # releases the top of each arena back to the kernel.
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass

    if DROP_PAGE_CACHE:
        # Drop only clean page cache (mode "1"): the weight-file bytes read during
        # load. Dirty pages are untouched, so this cannot lose data.
        try:
            with open("/proc/sys/vm/drop_caches", "w") as f:
                f.write("1")
        except OSError as e:
            logger.info("Skipping page-cache drop (need root / procfs): %s", e)


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


class _PhaseQueue:
    """Live queue-position accounting for one phase lock.

    ``enqueued`` dispenses a ticket to each request as it starts waiting for the
    lock; ``released`` counts requests that have left the wait (acquired-then-
    released, or abandoned it via cancel/disconnect). A waiter's queue_ahead is
    ``ticket - released``, which reaches 0 exactly when it is next in line. Both
    counters are only ever mutated from the single event-loop thread, so they
    need no locking."""

    def __init__(self) -> None:
        self.enqueued = 0
        self.released = 0


class _State:
    pipeline: Optional[object] = None
    # Texture-only pipeline (image + mesh -> textured mesh). Shares the resident
    # tex flow model / decoder / DINO / rembg with ``pipeline`` and additionally
    # holds a shape_slat_encoder. None if texturing could not be initialized.
    texturing_pipeline: Optional[object] = None
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
    # Per-lock queue accounting, so a client waiting for either phase gets a live
    # "queue_ahead" heartbeat instead of a silent stall (see _acquire_or_cancel).
    sampling_queue: "_PhaseQueue" = _PhaseQueue()
    postprocess_queue: "_PhaseQueue" = _PhaseQueue()
    ready: bool = False
    loaded_at: float = 0.0
    busy: bool = False
    rembg_warmup_status: str = "not_started"
    synthetic_warmup_status: str = "not_started"


state = _State()


class GenerationCancelled(Exception):
    """Raised by worker threads when their request has been cancelled."""


class _CancellationToken:
    """Cancellation state shared safely by the event loop and worker threads."""

    def __init__(self) -> None:
        self._thread_event = threading.Event()
        self._async_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        self._reason = "generation cancelled"
        self._reason_lock = threading.Lock()

    @property
    def reason(self) -> str:
        with self._reason_lock:
            return self._reason

    @property
    def cancelled(self) -> bool:
        return self._thread_event.is_set()

    def cancel(self, reason: str) -> None:
        with self._reason_lock:
            if self._thread_event.is_set():
                return
            self._reason = reason
            self._thread_event.set()
        try:
            self._loop.call_soon_threadsafe(self._async_event.set)
        except RuntimeError:
            # The server loop may already be shutting down. Worker-side checks
            # still see the threading event.
            pass

    async def wait(self) -> None:
        await self._async_event.wait()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise GenerationCancelled(self.reason)


@asynccontextmanager
async def _acquire_or_cancel(lock: asyncio.Lock, cancellation: _CancellationToken,
                             lock_name: str = "lock",
                             queue: Optional["_PhaseQueue"] = None,
                             queue_callback=None, queue_phase: str = "queued"):
    """Acquire a phase lock, but immediately remove cancelled queued work.

    Logs acquisition and release of the lock so GPU lock contention is
    observable in the server log.

    When ``queue`` is supplied the wait is accounted against that queue's live
    position counter, and if ``queue_callback`` (an async callable) is also
    given, a once-per-second heartbeat streams ``(queue_phase, waited_sec,
    queue_ahead)`` to the caller until the lock is acquired -- so a waiting
    client sees progress instead of a silent stall. The queue slot is freed
    (``queue.released``) once the request stops occupying the lock, whether it
    acquired-then-released it or abandoned the wait via cancellation.
    """
    cancellation.raise_if_cancelled()
    acquire_start = time.monotonic()

    heartbeat_task = None
    released_counted = False

    def _release_slot() -> None:
        nonlocal released_counted
        if queue is not None and not released_counted:
            released_counted = True
            queue.released += 1

    if queue is not None:
        my_ticket = queue.enqueued
        queue.enqueued += 1
        if queue_callback is not None:
            async def _heartbeat() -> None:
                while True:
                    await asyncio.sleep(1.0)
                    ahead = max(0, my_ticket - queue.released)
                    waited = round(time.monotonic() - acquire_start, 2)
                    await queue_callback(queue_phase, waited, ahead)
            heartbeat_task = asyncio.create_task(_heartbeat())

    acquire_task = asyncio.create_task(lock.acquire())
    cancel_task = asyncio.create_task(cancellation.wait())
    lock_held = False

    async def _stop_heartbeat() -> None:
        nonlocal heartbeat_task
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            heartbeat_task = None

    try:
        await asyncio.wait(
            (acquire_task, cancel_task), return_when=asyncio.FIRST_COMPLETED
        )
        if cancellation.cancelled:
            cancellation.raise_if_cancelled()

        await acquire_task
        lock_held = True
        await _stop_heartbeat()
        cancel_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_task
        logger.info("%s acquired after %.2fs", lock_name,
                    time.monotonic() - acquire_start)
        yield
    finally:
        await _stop_heartbeat()
        if not acquire_task.done():
            acquire_task.cancel()
        try:
            acquired = await acquire_task
        except asyncio.CancelledError:
            acquired = False
        if acquired and not lock_held:
            lock_held = True
        if not cancel_task.done():
            cancel_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_task
        if lock_held:
            logger.info("%s released (held=%.2fs)", lock_name,
                        time.monotonic() - acquire_start)
            lock.release()
        _release_slot()


async def _to_thread_cancellable(
    operation: Callable[..., T],
    *args,
    cancellation: _CancellationToken,
) -> T:
    """Keep a phase lock held until cancelled worker code has really stopped."""
    worker = asyncio.create_task(asyncio.to_thread(operation, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancellation.cancel("server request task cancelled")
        # asyncio cannot forcibly stop a thread. Wait for its cooperative
        # cancellation point before allowing another request onto the GPU.
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except GenerationCancelled:
                break
        if worker.done() and not worker.cancelled():
            with suppress(Exception):
                worker.result()
        raise


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
    texture_sampling_steps: int = 12             # tex SLat sampling steps
    shape_sampling_steps: int = 12               # shape SLat sampling steps
    tex_shape_slat: int = Field(default=512, ge=512, le=1024)  # mesh encoding grid resolution in texture mode (512~1024)
    alpha_mode: Literal['OPAQUE', 'MASK', 'BLEND'] = 'OPAQUE'  # OPAQUE / MASK / BLEND
    smooth_by_angle: Optional[float] = None       # None=off, float=angle threshold for edge splitting
    filename: Optional[str] = None                # original image filename (for logging)
    # Generation mode:
    #   full    - image -> textured GLB (shape + texture; default).
    #   mesh    - image -> white GLB (shape only, no UVs, no texture).
    #   texture - image + input GLB -> textured GLB (re-texture an existing
    #             mesh; the client sends the GLB as a second binary frame).
    #             Texture is sampled at 1024 (the only resident texture model).
    #   rmbg    - image -> PNG with background removed (no 3D generation).
    mode: Literal['full', 'mesh', 'texture', 'rmbg'] = 'full'


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

    # Hard-cap the torch allocator before any weight lands on the GPU so an
    # over-budget request raises a catchable CUDA OOM (one failed request)
    # instead of spilling into slow shared system memory or tripping the OS
    # OOM-killer. Applies to torch allocations only; must be set per process.
    if 0 < MEM_FRACTION < 1 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRACTION, torch.cuda.current_device())
        free, total = torch.cuda.mem_get_info()
        logger.info("torch allocator capped at %.0f%% of %.1fG VRAM",
                    MEM_FRACTION * 100, total / 1e9)

    logger.info("Loading pipeline from: %s; background-removal model: %s",
                MODEL_PATH, REMBG_MODEL_PATH or "pipeline default")
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
            rembg_model_path=REMBG_MODEL_PATH,
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
    _run_startup_stage("initializing texturing pipeline", _init_texturing_pipeline)
    if TRIM_MEMORY:
        _run_startup_stage("releasing transient host memory", _trim_host_memory)


# The texturing (image + mesh -> textured mesh) capability reuses the resident
# pipeline's tex flow model, tex decoder, image encoder and background remover;
# the only extra weight is the shape encoder, which the image-to-3D pipeline
# does not otherwise load. Set TRELLIS2_ENABLE_TEXTURING=0 to skip it entirely
# (texture-only requests then return an error) and avoid the extra VRAM.
ENABLE_TEXTURING = os.environ.get("TRELLIS2_ENABLE_TEXTURING", "1") == "1"
SHAPE_ENCODER_REF = os.environ.get(
    "TRELLIS2_SHAPE_ENCODER_REF", "ckpts/shape_enc_next_dc_f16c32_fp16"
)


def _init_texturing_pipeline() -> None:
    """Build a Trellis2TexturingPipeline that shares the resident models and adds
    a shape encoder. Failure is non-fatal: texture-only requests are rejected but
    image-to-3D (full / mesh-only) generation still works."""
    if not ENABLE_TEXTURING:
        logger.info("Texturing disabled (TRELLIS2_ENABLE_TEXTURING=0); "
                    "texture-only requests will be rejected")
        return
    try:
        from trellis2.pipelines import Trellis2TexturingPipeline
        from trellis2.pipelines.base import _load_model_from_local

        pipeline = state.pipeline
        encoder = _load_model_from_local(MODEL_PATH, SHAPE_ENCODER_REF, device="cuda")
        encoder.eval()

        tex = Trellis2TexturingPipeline.__new__(Trellis2TexturingPipeline)
        tex.models = {
            'shape_slat_encoder': encoder,
            'tex_slat_decoder': pipeline.models['tex_slat_decoder'],
            'tex_slat_flow_model_1024': pipeline.models['tex_slat_flow_model_1024'],
        }
        tex.tex_slat_sampler = pipeline.tex_slat_sampler
        tex.tex_slat_sampler_params = pipeline.tex_slat_sampler_params
        tex.shape_slat_normalization = pipeline.shape_slat_normalization
        tex.tex_slat_normalization = pipeline.tex_slat_normalization
        tex.image_cond_model = pipeline.image_cond_model
        tex.rembg_model = pipeline.rembg_model
        tex.pbr_attr_layout = pipeline.pbr_attr_layout
        tex._device = 'cuda'
        # Mirror the image-to-3D pipeline's offload state so texturing can page the
        # tex_slat_decoder to CPU during shape encoding (its VRAM-peak stage) and
        # bring it back only for decode. No-op when offload is disabled.
        tex._offload = bool(getattr(pipeline, '_offload', False))
        state.texturing_pipeline = tex
        logger.info("Texturing pipeline ready (shares resident tex models + "
                    "shape encoder '%s')", SHAPE_ENCODER_REF)
    except Exception:
        logger.exception("Texturing pipeline init failed; texture-only requests "
                         "will be rejected")
        state.texturing_pipeline = None


async def _background_rembg_warmup() -> None:
    """Warm RMBG immediately after startup in the background."""
    state.rembg_warmup_status = "running"
    try:
        rembg_model = getattr(state.pipeline, "rembg_model", None)
        if rembg_model is None:
            state.rembg_warmup_status = "skipped"
            logger.info("Background-removal warmup skipped: no model")
            return
        if getattr(rembg_model, "warmed", False):
            state.rembg_warmup_status = "complete"
            logger.info("Background-removal model was warmed by a request")
            return

        started_at = time.monotonic()
        logger.info("Background-removal warmup started source=%s",
                    getattr(rembg_model, "warmup_source", "generated image"))
        await asyncio.to_thread(rembg_model.warmup)
        state.rembg_warmup_status = "complete"
        logger.info("Background-removal warmup completed elapsed=%.2fs",
                    time.monotonic() - started_at)
    except asyncio.CancelledError:
        state.rembg_warmup_status = "cancelled"
        logger.info("Background-removal warmup cancelled")
        raise
    except Exception:
        state.rembg_warmup_status = "failed"
        logger.exception("Background-removal warmup failed")


def _make_synthetic_warmup_image() -> Image.Image:
    """Create a simple alpha-matted object image for startup warmup."""
    from PIL import ImageDraw

    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((116, 64, 396, 344), fill=(196, 82, 56, 255))
    draw.polygon(
        [(256, 352), (132, 452), (380, 452)],
        fill=(48, 104, 184, 255),
    )
    draw.rectangle((210, 252, 302, 420), fill=(236, 190, 72, 255))
    draw.ellipse((205, 140, 248, 183), fill=(255, 244, 224, 255))
    draw.ellipse((264, 140, 307, 183), fill=(255, 244, 224, 255))
    return image


async def _background_synthetic_request_warmup() -> None:
    """Run one synthetic generation after startup to JIT/autotune hot paths."""
    if not SYNTHETIC_WARMUP:
        state.synthetic_warmup_status = "skipped"
        logger.info("Synthetic request warmup disabled")
        return

    state.synthetic_warmup_status = "running"
    try:
        if SYNTHETIC_WARMUP_DELAY_SEC > 0:
            await asyncio.sleep(SYNTHETIC_WARMUP_DELAY_SEC)
        if not state.ready or state.pipeline is None:
            state.synthetic_warmup_status = "skipped"
            logger.info("Synthetic request warmup skipped: pipeline not ready")
            return

        params = GenParams(
            seed=SYNTHETIC_WARMUP_SEED,
            pipeline_type=SYNTHETIC_WARMUP_PIPELINE,
            texture_size=SYNTHETIC_WARMUP_TEXTURE_SIZE,
            decimation_target=SYNTHETIC_WARMUP_DECIMATION_TARGET,
            simplify=SYNTHETIC_WARMUP_SIMPLIFY,
            texture_sampling_steps=SYNTHETIC_WARMUP_TEXTURE_STEPS,
            shape_sampling_steps=SYNTHETIC_WARMUP_SHAPE_STEPS,
            filename="synthetic-warmup.png",
            mode=SYNTHETIC_WARMUP_MODE,
        )
        if params.mode == "texture":
            raise ValueError(
                "TRELLIS2_SYNTHETIC_WARMUP_MODE=texture needs an input mesh; "
                "use full, mesh, or rmbg"
            )

        image = _make_synthetic_warmup_image()
        request_id = f"warmup-{uuid.uuid4().hex[:6]}"
        started_at = time.monotonic()
        logger.info(
            "[%s] synthetic request warmup started mode=%s pipeline=%s "
            "shape_steps=%d texture_steps=%d texture_size=%d decimation=%d",
            request_id, params.mode, params.pipeline_type,
            params.shape_sampling_steps, params.texture_sampling_steps,
            params.texture_size, params.decimation_target,
        )
        if params.mode == "rmbg":
            data = await asyncio.to_thread(_run_rembg, image, "PNG")
        else:
            data = await _generate(image, params, request_id)
        state.synthetic_warmup_status = "complete"
        logger.info(
            "[%s] synthetic request warmup completed bytes=%d elapsed=%.2fs",
            request_id, len(data), time.monotonic() - started_at,
        )
    except asyncio.CancelledError:
        state.synthetic_warmup_status = "cancelled"
        logger.info("Synthetic request warmup cancelled")
        raise
    except Exception:
        state.synthetic_warmup_status = "failed"
        logger.exception("Synthetic request warmup failed")


async def _background_startup_warmups() -> None:
    """Run non-critical startup warmups without blocking server readiness."""
    await _background_rembg_warmup()
    await _background_synthetic_request_warmup()


class _ProgressReporter:
    """Per-request progress/timing, shared across the sampling and
    post-processing phases so the reported percentages and elapsed time stay
    continuous even though the two phases run under different locks (and
    possibly different worker threads)."""

    def __init__(self, request_id: str, cancellation: _CancellationToken,
                 progress_callback=None):
        self.request_id = request_id
        self.cancellation = cancellation
        self.progress_callback = progress_callback
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at
        self.last_logged_stage: Optional[str] = None

    def raise_if_cancelled(self) -> None:
        self.cancellation.raise_if_cancelled()

    def report(self, percent: int, stage: str) -> None:
        self.raise_if_cancelled()
        now = time.monotonic()
        elapsed = round(now - self.started_at, 2)
        # Log only the first report of each stage. Intra-stage updates (per
        # sampling step, per postprocess sub-step) still stream to the client but
        # would otherwise flood the log with near-identical lines. ``delta`` then
        # measures how long the previous stage took.
        if stage != self.last_logged_stage:
            delta = now - self.last_report_at
            self.last_report_at = now
            self.last_logged_stage = stage
            logger.info("[%s] progress=%d%% stage=%s elapsed=%.2fs delta=%.2fs",
                        self.request_id, percent, stage, elapsed, delta)
        if self.progress_callback is not None:
            self.progress_callback(percent, stage, elapsed)
        self.raise_if_cancelled()


def _ensure_tex_flow_resident() -> None:
    """Page the shared texture flow model back onto the GPU if CPU offload left
    it evicted. Texture-only sampling does not go through the offload-aware
    ``_resident`` context, so it must be resident before we sample.

    Called via the texturing pipeline's ``on_shape_encoded`` hook -- i.e. after
    shape encoding, the peak-VRAM stage -- so the flow model's weights are not
    resident during that peak (they would otherwise stack ~1.3B params on top of
    the shape encoder's activations)."""
    pipeline = state.pipeline
    if pipeline is None or not getattr(pipeline, "_offload", False):
        return
    model = pipeline.models.get('tex_slat_flow_model_1024')
    if model is None:
        return
    try:
        if next(model.parameters()).device.type != 'cuda':
            model.to('cuda')
            logger.info("paged tex flow model onto GPU for texturing")
    except StopIteration:
        pass


@torch.inference_mode()
def _run_texture_sampling(reporter: "_ProgressReporter", image: Image.Image,
                          input_mesh, params: GenParams):
    """GPU-heavy phase for texture-only mode: encode an existing mesh + image and
    sample/decode a texture, returning a fully textured trimesh."""
    tex_pipeline = state.texturing_pipeline
    if tex_pipeline is None:
        raise RuntimeError(
            "texture-only mode is unavailable (texturing pipeline not loaded)"
        )
    # The shape encoding grid resolution is controlled by ``tex_shape_slat``
    # (512 or 1024). Texture sampling always runs on tex_slat_flow_model_1024
    # (the dedicated 512 texture model was dropped); the pipeline conditions at
    # 1024 even when the shape is encoded at 512. The flow model is paged onto
    # the GPU via the on_shape_encoded hook (after shape encoding, the peak-VRAM
    # stage) rather than up front, so its weights don't inflate the encoder's
    # activation peak.
    def _evict_flow_before_decode() -> None:
        # Page the texture flow model back to CPU *before* decode_tex_slat, exactly
        # as the image-to-3D pipeline does before its own decode. The flow DiT is
        # idle during decode, so leaving it resident just inflates the decode peak
        # (the binding peak for texture-only). This also restores the offload
        # baseline for the next request.
        if getattr(state.pipeline, "_offload", False):
            state.pipeline._evict_flow_models()

    def pipeline_progress(percent: int, stage: str) -> None:
        # The texturing pipeline's 0..100 internal progress spans the sampling
        # phase's 15%..85% of the whole request (post-processing/export is the
        # remaining tail). Mapping it here lets the client stream progress through
        # shape encoding, per-step texture sampling and decode instead of stalling.
        reporter.report(15 + round(percent * 0.7), stage)

    out = tex_pipeline.run(
        input_mesh,
        image,
        seed=params.seed,
        preprocess_image=True,
        tex_slat_sampler_params={
            "steps": params.texture_sampling_steps,
            "guidance_strength": 4.0,
            "guidance_rescale": 0.2,
            "guidance_interval": [0.0, 0.9],
            "rescale_t": 3.0,
            "cancellation_callback": reporter.raise_if_cancelled,
        },
        resolution=params.tex_shape_slat,
        texture_size=params.texture_size,
        on_shape_encoded=_ensure_tex_flow_resident,
        before_decode=_evict_flow_before_decode,
        progress_callback=pipeline_progress,
    )
    return out


@torch.inference_mode()
def _run_sampling(reporter: "_ProgressReporter", image: Image.Image,
                  params: GenParams, input_mesh=None):
    """GPU-heavy phase: model sampling + VAE decode. Must run under the
    sampling lock. Returns the decoded mesh (GPU-resident) for full/mesh modes,
    or a textured trimesh for texture mode."""
    pipeline = state.pipeline
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if params.mode == 'texture':
        return _run_texture_sampling(reporter, image, input_mesh, params)

    def pipeline_progress(percent: int, stage: str) -> None:
        # The model pipeline is about 80% of the complete request; the rest is
        # mesh cleanup, GLB construction and file export.
        reporter.report(5 + round(percent * 0.8), stage)

    mesh = pipeline.run(
        image,
        seed=params.seed,
        pipeline_type=params.pipeline_type or DEFAULT_PIPELINE,
        max_num_tokens=params.max_num_tokens,
        max_active_tokens=MAX_ACTIVE_TOKENS,
        preprocess_image=True,
        generate_texture=params.mode != 'mesh',
        sparse_structure_sampler_params={
            "steps": params.shape_sampling_steps,
            "guidance_strength": 6.5,
            "guidance_rescale": 0.05,
            "guidance_interval": [0.1, 1.0],
            "rescale_t": 4.0,
            "cancellation_callback": reporter.raise_if_cancelled,
        },
        shape_slat_sampler_params={
            "steps": params.shape_sampling_steps,
            "guidance_strength": 6.5,
            "guidance_rescale": 0.05,
            "guidance_interval": [0.1, 1.0],
            "rescale_t": 4.0,
            "cancellation_callback": reporter.raise_if_cancelled,
        },
        tex_slat_sampler_params={
            "steps": params.texture_sampling_steps,
            "guidance_strength": 4.0,
            "guidance_rescale": 0.2,
            "guidance_interval": [0.0, 0.9],
            "rescale_t": 3.0,
            "cancellation_callback": reporter.raise_if_cancelled,
        },
        progress_callback=pipeline_progress,
    )[0]
    reporter.report(85, "simplifying mesh")
    mesh.simplify(params.simplify)
    return mesh


def _export_glb_bytes(glb, extension_webp: bool) -> bytes:
    """Export a trimesh to GLB bytes via a temp file (trimesh has no direct
    bytes export for GLB with the webp texture extension)."""
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        glb.export(tmp_path, extension_webp=extension_webp)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@torch.inference_mode()
def _run_postprocess(reporter: "_ProgressReporter", mesh, params: GenParams) -> bytes:
    """GPU-light phase: GLB texturing/geometry (cumesh) + export. Must run
    under the post-processing lock. This is the latency-bound tail that barely
    uses the GPU, so it is allowed to overlap the next request's sampling."""
    # Texture mode: sampling already produced a fully textured trimesh; just
    # export it. It carries a texture, so keep the fast WebP encoding.
    if params.mode == 'texture':
        reporter.report(90, "exporting textured GLB")
        data = _export_glb_bytes(mesh, extension_webp=True)
        reporter.report(100, f"complete ({len(data)} bytes)")
        return data

    geometry_only = params.mode == 'mesh'
    reporter.report(86, "building white mesh" if geometry_only
                    else "building GLB textures and materials")

    def postprocess_progress(percent: int, stage: str) -> None:
        reporter.report(86 + round(percent * 0.09), stage)

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
        alpha_mode=params.alpha_mode,
        smooth_by_angle=params.smooth_by_angle is not None,
        smooth_angle_deg=params.smooth_by_angle if params.smooth_by_angle is not None else 30.0,
        geometry_only=geometry_only,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=False,
        progress_callback=postprocess_progress,
    )

    reporter.report(95, "exporting GLB")
    # WebP texture encoding is markedly faster than PNG's single-threaded zlib
    # path and produces smaller GLBs (no-op for a white, texture-less mesh).
    # Requires Pillow built with WebP support (libwebp-dev) and a client/viewer
    # that understands the EXT_texture_webp glTF extension.
    data = _export_glb_bytes(glb, extension_webp=not geometry_only)
    reporter.report(100, f"complete ({len(data)} bytes)")
    return data


class _VramPeakMonitor:
    """Background sampler that logs each new VRAM high-water mark.

    Boundary probes miss a transient that is allocated *and* freed inside
    un-instrumented code (e.g. the pipeline decode). This polls the driver
    (``mem_get_info``, so it sees every allocator including raw cudaMalloc) on a
    fixed interval and logs whenever usage climbs by ``step_gb``. Correlate the
    log *timestamp* with the progress-stage logs to pin the peak to a stage.
    Gated by the shared OVOXEL_VRAM_LOG switch (set to 0 to disable).
    """

    def __init__(self, request_id: str, interval: float = 0.05,
                 step_gb: float = 0.5):
        self._request_id = request_id
        self._interval = interval
        self._step = step_gb
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._device = 0
        self.peak_gb = 0.0

    def _run(self) -> None:
        last_logged = 0.0
        while not self._stop.is_set():
            try:
                free, total = torch.cuda.mem_get_info(self._device)
            except Exception:
                break
            used = (total - free) / 1e9
            if used > self.peak_gb:
                self.peak_gb = used
            if used >= last_logged + self._step:
                last_logged = used
                logger.info("[%s] vram high-water: %.2fG", self._request_id, used)
            self._stop.wait(self._interval)

    def __enter__(self) -> "_VramPeakMonitor":
        if (os.environ.get("OVOXEL_VRAM_LOG", "1") != "0"
                and torch.cuda.is_available()):
            self._device = torch.cuda.current_device()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            logger.info("[%s] vram peak this request: %.2fG",
                        self._request_id, self.peak_gb)
        return False


async def _generate(image: Image.Image, params: GenParams, request_id: str,
                    progress_callback=None,
                    cancellation: Optional[_CancellationToken] = None,
                    input_mesh=None, queue_callback=None) -> bytes:
    """Two-phase image -> GLB.

    Phase 1 (sampling) holds the sampling lock and saturates the GPU. Phase 2
    (post-processing) holds only the post-processing lock, so once phase 1
    releases the sampling lock the next queued request can start sampling while
    this request finishes its GPU-light GLB construction and export.

    ``queue_callback`` (optional, async) is invoked once per second with
    ``(phase, waited_sec, queue_ahead)`` while this request is waiting for either
    phase lock, so a queued client gets a live heartbeat instead of a silent
    stall. It runs on the event loop, so it must be a coroutine function (unlike
    the thread-driven ``progress_callback``). Queue accounting and the heartbeat
    itself live in ``_acquire_or_cancel``.
    """
    cancellation = cancellation or _CancellationToken()
    reporter = _ProgressReporter(request_id, cancellation, progress_callback)

    with _VramPeakMonitor(request_id):
        async with _acquire_or_cancel(
            state.sampling_lock, cancellation,
            lock_name=f"[{request_id}] sampling_lock",
            queue=state.sampling_queue, queue_callback=queue_callback,
            queue_phase="sampling",
        ):
            state.busy = True
            try:
                mesh = await _to_thread_cancellable(
                    _run_sampling, reporter, image, params, input_mesh,
                    cancellation=cancellation,
                )
            finally:
                state.busy = False
            if not OVERLAP_POSTPROCESS:
                # Serial fallback: keep the sampling lock held through the whole
                # post-processing tail so only one request uses the GPU at a time.
                async with _acquire_or_cancel(
                    state.postprocess_lock, cancellation,
                    lock_name=f"[{request_id}] postprocess_lock",
                    queue=state.postprocess_queue, queue_callback=queue_callback,
                    queue_phase="postprocess",
                ):
                    data = await _to_thread_cancellable(
                        _run_postprocess, reporter, mesh, params,
                        cancellation=cancellation,
                    )
                    del mesh
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    _schedule_flow_prewarm()
                    return data

        # Overlap path: the sampling lock is released here; the next request can
        # sample while this one finishes post-processing under its own lock.
        async with _acquire_or_cancel(
            state.postprocess_lock, cancellation,
            lock_name=f"[{request_id}] postprocess_lock",
            queue=state.postprocess_queue, queue_callback=queue_callback,
            queue_phase="postprocess",
        ):
            data = await _to_thread_cancellable(
                _run_postprocess, reporter, mesh, params, cancellation=cancellation
            )
            del mesh
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _schedule_flow_prewarm()
            return data


# Strong refs to fire-and-forget prewarm tasks so the loop can't GC them mid-run.
_background_tasks: set = set()


def _schedule_flow_prewarm() -> None:
    """After a request finishes, kick off a background task that pages the flow
    models back onto the GPU for the next request ("decode" offload mode only).

    Fire-and-forget: it never delays returning this request's GLB, and the task
    itself skips the work whenever another request is already sampling, so it
    stays off every request's critical path.
    """
    pipeline = state.pipeline
    if pipeline is None or getattr(pipeline, "_offload_mode", None) != "decode":
        return
    task = asyncio.create_task(_prewarm_flow_models_when_idle())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _prewarm_flow_models_when_idle() -> None:
    """Warm the flow models back onto the GPU while the GPU is idle.

    Concurrency: every flow-model device movement (page-in during sampling,
    eviction before decode, and this prewarm) must be mutually exclusive, so we
    do the prewarm under ``sampling_lock`` -- the same lock that guards
    generation. If a request is already sampling we skip entirely rather than
    queue: it pages in what it needs itself, and warming behind it would just
    contend for the GPU. The event loop is single-threaded, so a ``False`` from
    ``locked()`` guarantees the ``async with`` below takes the lock immediately
    without yielding -- no request can slip in between the check and the acquire.
    """
    pipeline = state.pipeline
    if pipeline is None or state.sampling_lock.locked():
        return
    async with state.sampling_lock:
        if state.pipeline is None:
            return
        try:
            await asyncio.to_thread(pipeline.prewarm_flow_models)
        except Exception:
            logger.exception("flow-model prewarm failed")


def _run_rembg(image: Image.Image, output_format: str = "PNG") -> bytes:
    """Run background removal only and return the cutout encoded in
    ``output_format`` ("PNG" or "WEBP"). Both formats carry the alpha channel;
    WebP is saved losslessly so the cutout matte is preserved exactly."""
    rembg_model = state.pipeline.rembg_model
    if rembg_model is None:
        raise RuntimeError("Background removal model is not loaded")
    result = rembg_model(image)
    buf = io.BytesIO()
    if output_format == "WEBP":
        result.save(buf, format="WEBP", lossless=True)
    else:
        result.save(buf, format="PNG")
    return buf.getvalue()


def _decode_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def _decode_mesh(data: bytes):
    """Load an uploaded GLB (texture-only mode) into a single trimesh."""
    import trimesh
    loaded = trimesh.load(io.BytesIO(data), file_type="glb", process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) == 0:
            raise ValueError("no geometry found in uploaded GLB")
        loaded = loaded.to_geometry()
    if not hasattr(loaded, "vertices") or not hasattr(loaded, "faces"):
        raise ValueError("uploaded file did not contain a triangle mesh")
    return loaded


async def _watch_ws_cancellation(
    ws: WebSocket, cancellation: _CancellationToken, request_id: str
) -> bool:
    """Return True for an explicit cancel message, False for a disconnect."""
    try:
        while True:
            message = await ws.receive_json()
            message_type = message.get("type", message.get("action", ""))
            if str(message_type).lower() in {"cancel", "interrupt"}:
                logger.info("[%s] explicit cancellation requested", request_id)
                cancellation.cancel("client requested cancellation")
                return True
            logger.warning("[%s] ignoring WebSocket message type=%r",
                           request_id, message_type)
    except WebSocketDisconnect:
        logger.warning("[%s] WebSocket disconnected; cancelling generation",
                       request_id)
        cancellation.cancel("WebSocket client disconnected")
        return False
    except RuntimeError:
        # Starlette raises RuntimeError when receive() observes a connection
        # that has already transitioned to the disconnected state.
        cancellation.cancel("WebSocket client disconnected")
        return False


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load synchronously so /health only reports ready once the model is up.
    await asyncio.to_thread(_load_pipeline)
    warmup_task = asyncio.create_task(_background_startup_warmups())
    try:
        yield
    finally:
        warmup_task.cancel()
        with suppress(asyncio.CancelledError):
            await warmup_task
        state.pipeline = None
        state.ready = False


app = FastAPI(title="TRELLIS.2 Inference Server", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    if not state.ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {
        "status": "ok",
        "busy": state.busy,
        "rembg_warmup": state.rembg_warmup_status,
        "synthetic_warmup": state.synthetic_warmup_status,
    }


@app.websocket("/ws/generate")
async def ws_generate(ws: WebSocket):
    """
    Protocol:
      client -> {"seed": 42, "mode": "full"|"mesh"|"texture"|"rmbg", ...params}  (JSON text frame)
      client -> <raw image bytes>               (binary frame)
      client -> <raw GLB bytes>                 (binary frame; only when mode="texture")
      client -> {"type": "cancel"}               (while generation is running)
      server -> {"stage": "queued"|"processing"|"done"|"cancelled"|"error", ...}
      server -> {"stage": "done", "glb_size": N, ...}  (JSON text frame)
                (rmbg mode also sets "format": "png"|"webp" on the done frame)
      server -> <raw GLB bytes>                 (binary frame)
      server -> <raw image bytes>               (binary frame; only when mode="rmbg";
                                                 PNG in -> PNG out, otherwise WebP out)
    """
    request_id = uuid.uuid4().hex[:8]
    cancellation = None
    generation_task = None
    receiver_task = None
    await ws.accept()
    _ws_started_at = time.monotonic()
    logger.info("[%s] WebSocket connected client=%s", request_id, ws.client)
    try:
        try:
            if time.monotonic() - _ws_started_at >= WS_TOTAL_TIMEOUT:
                raise asyncio.TimeoutError("total connection timeout")
            raw = await asyncio.wait_for(ws.receive_text(), timeout=RECV_IDLE_TIMEOUT)
            req = json.loads(raw)
        except asyncio.TimeoutError:
            logger.warning("[%s] receive params timed out", request_id)
            await ws.send_json({"stage": "error", "message": "receive params timed out"})
            await ws.close()
            return
        except Exception as e:
            logger.warning("[%s] invalid params: %s", request_id, e)
            await ws.send_json({"stage": "error", "message": f"invalid params: {e}"})
            await ws.close()
            return
        logger.info("[%s] WebSocket generation request received", request_id)
        if not state.ready:
            logger.warning("[%s] rejected: model still loading", request_id)
            await ws.send_json({"stage": "error", "message": "model still loading"})
            await ws.close()
            return
        params = GenParams(**{k: v for k, v in req.items() if k in GenParams.model_fields})

        # rmbg mode: only remove background, return PNG. No pipeline needed.
        if params.mode == 'rmbg':
            try:
                if time.monotonic() - _ws_started_at >= WS_TOTAL_TIMEOUT:
                    raise asyncio.TimeoutError("total connection timeout")
                image_data = await asyncio.wait_for(ws.receive_bytes(), timeout=RECV_IDLE_TIMEOUT)
                img = _decode_image(image_data)
            except asyncio.TimeoutError:
                logger.warning("[%s] rmbg receive timed out", request_id)
                await ws.send_json({"stage": "error", "message": "receive timed out"})
                await ws.close()
                return
            except Exception as e:
                logger.warning("[%s] invalid image: %s", request_id, e)
                await ws.send_json({"stage": "error", "message": f"invalid image: {e}"})
                await ws.close()
                return

            # PNG in -> PNG out; every other input format (WebP, JPEG, ...) ->
            # WebP out. Both carry alpha for the cutout.
            input_format = (img.format or "PNG").upper()
            output_format = "PNG" if input_format == "PNG" else "WEBP"
            logger.info("[%s] rmbg request image=%s size=%sx%s mode=%s in=%s out=%s",
                        request_id, len(image_data), img.width, img.height, img.mode,
                        input_format, output_format)

            await ws.send_json({
                "stage": "queued", "queued": False, "request_id": request_id
            })
            await ws.send_json({"stage": "processing", "progress": 0, "request_id": request_id})

            t0 = time.time()
            out_bytes = await asyncio.to_thread(_run_rembg, img, output_format)

            await ws.send_json({
                "stage": "done",
                "elapsed_sec": round(time.time() - t0, 2),
                "progress": 100,
                "request_id": request_id,
                "glb_size": len(out_bytes),
                "format": output_format.lower(),
            })
            await ws.send_bytes(out_bytes)
            logger.info("[%s] rmbg request completed bytes=%d format=%s elapsed=%.2fs",
                        request_id, len(out_bytes), output_format, time.time() - t0)
            await ws.close()
            return

        if params.mode == 'texture' and state.texturing_pipeline is None:
            logger.warning("[%s] rejected: texture mode unavailable", request_id)
            await ws.send_json({"stage": "error",
                                "message": "texture-only mode is not enabled on this server"})
            await ws.close()
            return
        try:
            if time.monotonic() - _ws_started_at >= WS_TOTAL_TIMEOUT:
                raise asyncio.TimeoutError("total connection timeout")
            image_data = await asyncio.wait_for(ws.receive_bytes(), timeout=RECV_IDLE_TIMEOUT)
            img = _decode_image(image_data)
        except asyncio.TimeoutError:
            logger.warning("[%s] image receive timed out", request_id)
            await ws.send_json({"stage": "error", "message": "image receive timed out"})
            await ws.close()
            return
        except Exception as e:
            logger.warning("[%s] invalid image: %s", request_id, e)
            await ws.send_json({"stage": "error", "message": f"invalid image: {e}"})
            await ws.close()
            return

        input_mesh = None
        if params.mode == 'texture':
            try:
                if time.monotonic() - _ws_started_at >= WS_TOTAL_TIMEOUT:
                    raise asyncio.TimeoutError("total connection timeout")
                mesh_data = await asyncio.wait_for(ws.receive_bytes(), timeout=RECV_IDLE_TIMEOUT)
                input_mesh = _decode_mesh(mesh_data)
            except asyncio.TimeoutError:
                logger.warning("[%s] mesh receive timed out", request_id)
                await ws.send_json({"stage": "error", "message": "mesh receive timed out"})
                await ws.close()
                return
            except Exception as e:
                logger.warning("[%s] invalid mesh: %s", request_id, e)
                await ws.send_json({"stage": "error", "message": f"invalid mesh: {e}"})
                await ws.close()
                return
            logger.info("[%s] mesh decoded bytes=%d verts=%d faces=%d",
                        request_id, len(mesh_data),
                        len(input_mesh.vertices), len(input_mesh.faces))

        logger.info("[%s] image decoded bytes=%d size=%sx%s mode=%s filename=%s params=%s",
                    request_id, len(image_data), img.width, img.height, img.mode,
                    params.filename or "(unknown)", params.model_dump())

        loop = asyncio.get_running_loop()
        cancellation = _CancellationToken()

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
                future.cancel()
                cancellation.cancel("WebSocket client disconnected")

        async def send_queue(phase, waited_sec, queue_ahead):
            # Runs on the event loop (from the heartbeat task in _acquire_or_cancel),
            # so send directly rather than round-tripping through
            # run_coroutine_threadsafe. ``phase`` is "sampling" (before generation)
            # or "postprocess" (mesh already sampled, waiting for the export slot).
            try:
                await ws.send_json({
                    "stage": "queued", "phase": phase, "queued": True,
                    "waited_sec": waited_sec, "queue_ahead": queue_ahead,
                    "request_id": request_id,
                })
            except Exception:
                cancellation.cancel("WebSocket client disconnected")

        queued = state.sampling_lock.locked()
        await ws.send_json({
            "stage": "queued", "queued": queued, "request_id": request_id
        })
        await ws.send_json({"stage": "processing", "progress": 0, "request_id": request_id})

        t0 = time.time()
        # Locking (and the sampling/post-processing phase split) lives in
        # _generate so HTTP and WebSocket share one scheduling path.
        generation_task = asyncio.create_task(
            _generate(
                img, params, request_id, send_progress,
                cancellation=cancellation, input_mesh=input_mesh,
                queue_callback=send_queue,
            )
        )
        receiver_task = asyncio.create_task(
            _watch_ws_cancellation(ws, cancellation, request_id)
        )
        done, _ = await asyncio.wait(
            (generation_task, receiver_task), return_when=asyncio.FIRST_COMPLETED
        )

        explicit_cancel = False
        if receiver_task in done:
            explicit_cancel = receiver_task.result()

        if cancellation.cancelled:
            try:
                await generation_task
            except GenerationCancelled:
                pass
            if explicit_cancel:
                await ws.send_json({
                    "stage": "cancelled",
                    "message": cancellation.reason,
                    "request_id": request_id,
                })
                await ws.close()
            logger.info("[%s] WebSocket generation cancelled: %s",
                        request_id, cancellation.reason)
            return

        receiver_task.cancel()
        with suppress(asyncio.CancelledError):
            await receiver_task
        glb = await generation_task
        await ws.send_json({
            "stage": "done",
            "elapsed_sec": round(time.time() - t0, 2),
            "progress": 100,
            "request_id": request_id,
            "glb_size": len(glb),
        })
        await ws.send_bytes(glb)
        logger.info("[%s] WebSocket request completed bytes=%d elapsed=%.2fs",
                    request_id, len(glb), time.time() - t0)
        await ws.close()
    except WebSocketDisconnect:
        logger.warning("[%s] WebSocket disconnected", request_id)
    except GenerationCancelled as e:
        logger.info("[%s] WebSocket generation cancelled: %s", request_id, e)
    except Exception as e:
        # A token-limit rejection is an expected, predicted outcome -- log it as a
        # warning without the alarming full traceback the generic path prints.
        from trellis2.pipelines import ActiveTokenLimitExceeded
        if isinstance(e, ActiveTokenLimitExceeded):
            logger.warning("[%s] rejected: %s", request_id, e)
        else:
            logger.exception("[%s] WebSocket generation failed", request_id)
        try:
            await ws.send_json({"stage": "error", "message": str(e)})
            await ws.close()
        except Exception:
            pass
    finally:
        if receiver_task is not None and not receiver_task.done():
            receiver_task.cancel()
            with suppress(asyncio.CancelledError):
                await receiver_task
        if generation_task is not None and not generation_task.done():
            cancellation.cancel("WebSocket handler stopped")
            with suppress(asyncio.CancelledError, GenerationCancelled, Exception):
                await asyncio.shield(generation_task)


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
