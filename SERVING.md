# TRELLIS.2 Service Mode (vLLM-style)

Run TRELLIS.2 as a persistent inference server. The model pipeline is loaded
**once at startup and kept resident in RAM/VRAM**; clients submit an image over
HTTP or WebSocket and get back a textured **GLB** (with baked PBR texture). No
video is rendered.

Generations are serialized through a single-GPU work queue: the server accepts
many concurrent connections but runs one job at a time.

## Start the server

### Docker (recommended)

```bash
# docker-compose: default command is now `serve`
docker compose up

# or the Windows helper
docker_run.bat serve
```

The server is reachable on the host at **http://localhost:8086** (mapped to the
container's internal port 8000).

### Bare metal

```bash
python serve.py --host 0.0.0.0 --port 8000
```

### Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `TRELLIS2_MODEL_PATH` | `/models/microsoft/TRELLIS.2-4B` | Weights path / HF repo |
| `TRELLIS2_PIPELINE` | `512` | Default pipeline type: `512` / `1024` / `1024_cascade` / `1536_cascade` |
| `TRELLIS2_PORT` | `8000` | Listen port |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `200 {"status":"ok"}` when ready, `503` while loading |
| `GET` | `/info` | Model path, device, busy state |
| `POST` | `/generate` | multipart `image` upload → binary GLB |
| `WS` | `/ws/generate` | streams `queued`/`processing`/`done` JSON with `progress` and `step`; supports cancellation; final message carries `glb_size` (JSON text frame) followed by raw GLB bytes (binary frame) |

Generation params (form fields / JSON keys): `seed`, `pipeline_type`,
`texture_size` (default 2048), `decimation_target` (default 100000),
`max_num_tokens` (default 49152).
Image preprocessing is always enabled.
Remeshing and source-mesh attribute projection are always enabled because both
are required for acceptable mesh quality.

## Client

```bash
# WebSocket (streams progress)
python trellis2_client.py --image assets/example_image/T.png --output out.glb
python trellis2_client.py --image in.png --output out.glb \
    --server http://localhost:8086 --pipeline-type 512
```

### curl

```bash
# health
curl http://localhost:8086/health

# generate -> save GLB
curl -X POST http://localhost:8086/generate \
    -F image=@assets/example_image/T.png \
    -F texture_size=2048 \
    -o out.glb
```

## Logs and progress

Service logs include millisecond timestamps and a short request ID. Each
generation reports queue wait time and milestone progress from image
preprocessing through GLB export, followed by per-stage timings and CUDA peak
allocated/reserved memory. Follow them with:

```bash
docker logs --follow trellis2
```

The client uses WebSocket and displays generation progress in real time. The
HTTP endpoint remains available for integrations such as `curl`. Disconnecting
either an HTTP or WebSocket client cancels its queued/running generation. A
WebSocket client can also cancel explicitly while the job is running:

```json
{"type": "cancel"}
```

The server confirms an explicit request with a `{"stage": "cancelled", ...}`
message. `trellis2_client.py` sends this message automatically on Ctrl+C.
