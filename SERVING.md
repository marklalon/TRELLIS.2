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
| `TRELLIS2_LOW_VRAM` | `1` | `1` = offload submodules to CPU between steps; `0` = keep whole pipeline resident in VRAM (faster, needs more memory) |
| `TRELLIS2_PIPELINE` | `1024_cascade` | Default pipeline type: `512` / `1024` / `1024_cascade` / `1536_cascade` |
| `TRELLIS2_PORT` | `8000` | Listen port |

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `200 {"status":"ok"}` when ready, `503` while loading |
| `GET` | `/info` | Model path, device, busy state |
| `POST` | `/generate` | multipart `image` upload → binary GLB |
| `WS` | `/ws/generate` | streams `queued`/`processing`/`done` JSON; final message carries `glb_base64` |

Generation params (form fields / JSON keys): `seed`, `pipeline_type`,
`texture_size` (default 4096), `decimation_target` (default 1000000),
`max_num_tokens` (default 49152), `preprocess_image` (default true).

## Client

```bash
# HTTP
python trellis2_client.py --image assets/example_image/T.png --output out.glb

# WebSocket (streams progress)
python trellis2_client.py --image in.png --output out.glb --ws \
    --server http://localhost:8086 --pipeline-type 1024_cascade
```

### curl

```bash
# health
curl http://localhost:8086/health

# generate -> save GLB
curl -X POST http://localhost:8086/generate \
    -F image=@assets/example_image/T.png \
    -F texture_size=4096 \
    -o out.glb
```
