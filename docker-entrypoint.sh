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

exec python serve.py --host "${TRELLIS2_HOST:-0.0.0.0}" --port "${TRELLIS2_PORT:-8000}"
