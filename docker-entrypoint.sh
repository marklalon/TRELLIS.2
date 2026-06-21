#!/bin/bash
# Docker entrypoint — provides convenient shortcuts for TRELLIS.2
set -e

echo "============================================"
echo "  TRELLIS.2 Docker Container"
echo "  Model path: ${TRELLIS2_MODEL_PATH:-/models/microsoft/TRELLIS.2-4B}"
echo "============================================"

case "${1:-demo}" in
    demo)
        echo "Running image-to-3D demo..."
        exec python docker_demo.py
        ;;
    app)
        echo "Launching Gradio web UI on port 7860..."
        exec python app.py
        ;;
    texturing)
        echo "Running texturing demo..."
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
