#!/bin/bash
# Docker entrypoint — provides convenient shortcuts for TRELLIS.2
set -e

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S.%3N')" "$*"
}

log "============================================"
log "  TRELLIS.2 Docker Container"
log "  Model path: ${TRELLIS2_MODEL_PATH:-/models/microsoft/TRELLIS.2-4B}"
log "============================================"

case "${1:-demo}" in
    demo)
        log "Running image-to-3D demo..."
        exec python docker_demo.py
        ;;
    app)
        log "Launching Gradio web UI on port 7860..."
        exec python app.py
        ;;
    serve)
        log "Launching TRELLIS.2 inference server on port ${TRELLIS2_PORT:-8000}..."
        exec python serve.py --host "${TRELLIS2_HOST:-0.0.0.0}" --port "${TRELLIS2_PORT:-8000}"
        ;;
    texturing)
        log "Running texturing demo..."
        exec python example_texturing.py
        ;;
    bash|shell)
        exec /bin/bash
        ;;
    python)
        shift
        exec python "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
