"""
TRELLIS.2 service client — send an image, save a textured GLB.

HTTP (default):
    python trellis2_client.py --image assets/example_image/T.png --output out.glb
    python trellis2_client.py --image in.png --server http://HOST:8000 --pipeline-type 1024_cascade

WebSocket (streams progress):
    python trellis2_client.py --image in.png --output out.glb --ws

Only needs `requests` for HTTP, and `websockets` for the --ws mode.
"""
import argparse
import base64
import sys
import time


def run_http(args):
    import requests
    url = args.server.rstrip("/") + "/generate"
    data = {
        "seed": args.seed,
        "texture_size": args.texture_size,
        "decimation_target": args.decimation_target,
        "preprocess_image": str(args.preprocess).lower(),
    }
    if args.pipeline_type:
        data["pipeline_type"] = args.pipeline_type
    started_at = time.monotonic()
    try:
        with open(args.image, "rb") as f:
            files = {"image": (args.image, f, "application/octet-stream")}
            print(f"[client] POST {url}")
            resp = requests.post(url, data=data, files=files, timeout=args.timeout)
        if resp.status_code != 200:
            print(f"[client] error {resp.status_code}: {resp.text}", file=sys.stderr)
            sys.exit(1)
        with open(args.output, "wb") as f:
            f.write(resp.content)
        print(f"[client] saved {len(resp.content)} bytes -> {args.output}")
    finally:
        print(f"[client] request elapsed: {time.monotonic() - started_at:.2f}s")


def run_ws(args):
    import asyncio
    import json
    import websockets

    async def _go():
        ws_url = args.server.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
        ws_url += "/ws/generate"
        with open(args.image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "image_base64": image_b64,
            "seed": args.seed,
            "texture_size": args.texture_size,
            "decimation_target": args.decimation_target,
            "preprocess_image": args.preprocess,
        }
        if args.pipeline_type:
            payload["pipeline_type"] = args.pipeline_type
        print(f"[client] connecting {ws_url}")
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            started_at = time.monotonic()
            try:
                await ws.send(json.dumps(payload))
                while True:
                    msg = json.loads(await ws.recv())
                    stage = msg.get("stage")
                    if stage == "done":
                        glb = base64.b64decode(msg["glb_base64"])
                        with open(args.output, "wb") as f:
                            f.write(glb)
                        print(f"[client] done in {msg.get('elapsed_sec')}s, "
                              f"saved {len(glb)} bytes -> {args.output}")
                        break
                    elif stage == "error":
                        print(f"[client] server error: {msg.get('message')}", file=sys.stderr)
                        sys.exit(1)
                    else:
                        print(f"[client] {stage} {({k: v for k, v in msg.items() if k != 'stage'})}")
            finally:
                print(f"[client] request elapsed: {time.monotonic() - started_at:.2f}s")

    asyncio.run(_go())


def main():
    p = argparse.ArgumentParser(description="TRELLIS.2 service client")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--output", default="outputs/output.glb", help="Output GLB path")
    p.add_argument("--server", default="http://localhost:8086", help="Server base URL")
    p.add_argument("--ws", action="store_true", help="Use WebSocket (streams progress)")
    p.add_argument("--pipeline-type", default=None,
                   help="512 / 1024 / 1024_cascade / 1536_cascade (default: server's)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--texture-size", type=int, default=2048)
    p.add_argument("--decimation-target", type=int, default=100000)
    p.add_argument("--preprocess", type=lambda s: s.lower() not in ("0", "false", "no"),
                   default=True, help="Preprocess/remove background (default: true)")
    p.add_argument("--timeout", type=int, default=1800, help="HTTP timeout seconds")
    args = p.parse_args()

    if args.ws:
        run_ws(args)
    else:
        run_http(args)


if __name__ == "__main__":
    main()
