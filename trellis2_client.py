"""TRELLIS.2 WebSocket client — stream progress and save a GLB or PNG.

Modes:
    full     image -> textured GLB (default)
    mesh     image -> white GLB (geometry only, no UVs / texture)
    texture  image + input mesh -> textured GLB (re-texture an existing mesh)
    rmbg     image -> PNG with background removed (no 3D generation)

Example:
    python trellis2_client.py --image assets/example_image/T.png --output out.glb
    python trellis2_client.py --image in.png --mode mesh --output white.glb
    python trellis2_client.py --image in.png --mode texture --input-mesh model.glb --output textured.glb
    python trellis2_client.py --image in.png --mode rmbg --output out.png

Requires the ``websockets`` package.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import suppress

def _websocket_url(server: str) -> str:
    base = server.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    elif not base.startswith(("ws://", "wss://")):
        base = "ws://" + base
    return base + "/ws/generate"


class ProgressDisplay:
    """Render progress in place on a terminal, or as lines when redirected."""

    def __init__(self, width: int = 30):
        self.width = width
        self.last_length = 0
        self.active = False

    def update(self, percent: int, step: str, elapsed: float | None = None) -> None:
        percent = max(0, min(100, int(percent)))
        filled = round(self.width * percent / 100)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed_text = f"  {elapsed:.1f}s" if elapsed is not None else ""
        line = f"[client] [{bar}] {percent:3d}%  {step}{elapsed_text}"

        if sys.stdout.isatty():
            sys.stdout.write("\r" + line.ljust(self.last_length))
            sys.stdout.flush()
            self.last_length = max(self.last_length, len(line))
            self.active = True
        else:
            print(line)

    def finish(self) -> None:
        if self.active:
            print()
            self.active = False


async def _run(args) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency 'websockets'; install it with: pip install websockets"
        ) from exc

    ws_url = _websocket_url(args.server)
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    mesh_bytes = None
    if args.mode == "texture":
        if not args.input_mesh:
            raise RuntimeError("--input-mesh is required when --mode texture")
        with open(args.input_mesh, "rb") as f:
            mesh_bytes = f.read()

    payload = {
        "seed": args.seed,
        "mode": args.mode,
        "texture_size": args.texture_size,
        "decimation_target": args.decimation_target,
        "simplify": args.simplify,
        "texture_sampling_steps": args.texture_sampling_steps,
        "shape_sampling_steps": args.shape_sampling_steps,
        "alpha_mode": args.alpha_mode,
        "smooth_by_angle": args.smooth_by_angle,
    }
    payload["pipeline_type"] = args.pipeline_type
    payload["tex_shape_slat"] = args.tex_shape_slat

    progress = ProgressDisplay()
    started_at = time.monotonic()
    print(f"[client] connecting {ws_url}")
    try:
        async with websockets.connect(
            ws_url,
            max_size=64 * 1024 * 1024,
            open_timeout=args.timeout,
        ) as ws:
            await ws.send(json.dumps(payload))
            await ws.send(image_bytes)
            if mesh_bytes is not None:
                await ws.send(mesh_bytes)

            try:
                async for raw_message in ws:
                    message = json.loads(raw_message)
                    stage = message.get("stage", "unknown")

                    if stage == "queued":
                        if message.get("queued"):
                            print("[client] queued; waiting for the GPU")
                        else:
                            print("[client] GPU is available; starting generation")
                    elif stage == "processing":
                        progress.update(
                            message.get("progress", 0),
                            message.get("step", "processing"),
                            message.get("elapsed_sec"),
                        )
                    elif stage == "done":
                        progress.update(100, "complete", message.get("elapsed_sec"))
                        progress.finish()
                        data = await ws.recv()
                        output_dir = os.path.dirname(os.path.abspath(args.output))
                        os.makedirs(output_dir, exist_ok=True)
                        if args.mode == "rmbg":
                            # rmbg echoes the input image format: PNG in -> PNG
                            # out, WebP in -> WebP out. Match the output file
                            # extension to the format the server reports.
                            fmt = str(message.get("format", "png")).lower()
                            ext = "." + fmt
                            output_path = args.output
                            if not output_path.lower().endswith(ext):
                                output_path = os.path.splitext(output_path)[0] + ext
                        else:
                            output_path = args.output
                        with open(output_path, "wb") as f:
                            f.write(data)
                        print(f"[client] saved {len(data)} bytes -> {output_path}")
                        return
                    elif stage == "cancelled":
                        raise asyncio.CancelledError(message.get("message"))
                    elif stage == "error":
                        raise RuntimeError(message.get("message", "unknown server error"))
                    else:
                        print(f"[client] server stage: {stage}")
            except asyncio.CancelledError:
                # asyncio.run turns Ctrl+C into task cancellation first. Send
                # the protocol-level message before the context closes so the
                # server can stop without waiting for disconnect detection.
                with suppress(Exception):
                    await asyncio.shield(ws.send(json.dumps({"type": "cancel"})))
                raise

            raise RuntimeError("WebSocket closed before the result was received")
    finally:
        progress.finish()
        print(f"[client] request elapsed: {time.monotonic() - started_at:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRELLIS.2 WebSocket client with live progress"
    )
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="outputs/output.glb", help="Output GLB path")
    parser.add_argument("--server", default="http://localhost:8086", help="Server base URL")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "mesh", "texture", "rmbg"],
        help="full: textured GLB (default); mesh: white geometry only; "
             "texture: re-texture --input-mesh with --image; "
             "rmbg: remove background, return PNG",
    )
    parser.add_argument(
        "--input-mesh",
        default=None,
        help="Input GLB path (required for --mode texture)",
    )
    # Kept so existing commands using --ws continue to work; WebSocket is now
    # always enabled.
    parser.add_argument("--ws", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--pipeline-type",
        default="1024_cascade",
        help="512 / 1024 / 1024_cascade / 1536_cascade (default: 1024_cascade)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--decimation-target", type=int, default=100000)
    parser.add_argument(
        "--texture-sampling-steps",
        type=int,
        default=12,
        help="Texture SLat sampling steps (default: 12)",
    )
    parser.add_argument(
        "--shape-sampling-steps",
        type=int,
        default=12,
        help="Shape SLat sampling steps (default: 12)",
    )
    parser.add_argument(
        "--tex-shape-slat",
        type=int,
        default=512,
        choices=[512, 1024],
        help="Mesh shape encoding grid resolution in texture mode (512 or 1024; default: 512)",
    )
    parser.add_argument(
        "--simplify",
        type=int,
        default=5000000,
        help="Mesh simplify target vertex count / nvdiffrast limit (16777216)",
    )
    parser.add_argument(
        "--alpha-mode",
        type=str,
        default='OPAQUE',
        choices=['OPAQUE', 'MASK', 'BLEND'],
        help="Alpha mode for PBR material (default: OPAQUE)",
    )
    parser.add_argument(
        "--smooth-by-angle",
        nargs="?",
        const=30.0,
        type=float,
        default=None,
        help="Split hard edges before UV unwrap (default angle: 30 deg). "
             "Usage: --smooth-by-angle (uses 30 deg) or --smooth-by-angle 45. "
             "Omit to disable smoothing.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Connection timeout seconds")
    args = parser.parse_args()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n[client] cancelled", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[client] error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
