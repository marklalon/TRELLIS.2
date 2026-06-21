# TRELLIS.2 Docker Image
# Base: CUDA 12.8.1 + cuDNN devel (Blackwell sm_120)
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="12.0" \
    MAX_JOBS=4

# ---- System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    libjpeg-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && ln -sf python3.10 /usr/bin/python \
    && ln -sf python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip ----
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# ---- PyTorch 2.7.1 + CUDA 12.8 ----
RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ---- Basic Python dependencies ----
RUN pip install --no-cache-dir \
    imageio==2.37.3 \
    imageio-ffmpeg==0.6.0 \
    tqdm==4.68.3 \
    easydict==1.13 \
    opencv-python-headless==4.13.0.92 \
    ninja==1.13.0 \
    trimesh==4.12.2 \
    transformers==5.12.1 \
    gradio==6.0.1 \
    tensorboard==2.20.0 \
    pandas==2.3.3 \
    lpips==0.1.4 \
    zstandard==0.25.0 \
    huggingface_hub==1.20.1 \
    safetensors==0.8.0 \
    plyfile==1.1.4 \
    kornia==0.8.2 \
    timm==1.0.27 \
    scipy==1.15.3 \
    pyyaml==6.0.3

# ---- utils3d from GitHub ----
RUN pip install --no-cache-dir \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# ---- pillow-simd ----
RUN pip install --no-cache-dir pillow-simd

# ---- Flash Attention 2.7.4.post1 (supports Blackwell sm_120) ----
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# ---- nvdiffrast ----
RUN mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast --no-build-isolation --no-cache-dir

# ---- nvdiffrec (renderutils branch, pinned commit) ----
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    git -C /tmp/extensions/nvdiffrec checkout b296927cc7fd01c2ac1087c8065c4d7248f72da4 && \
    pip install /tmp/extensions/nvdiffrec --no-build-isolation --no-cache-dir

# ---- CuMesh (pinned commit) ----
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh && \
    git -C /tmp/extensions/CuMesh checkout 12289e1062f0603f2f0d0771b02e1395d247f26f && \
    git -C /tmp/extensions/CuMesh submodule update --init --recursive && \
    pip install /tmp/extensions/CuMesh --no-build-isolation --no-cache-dir

# ---- FlexGEMM (pinned commit) ----
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM && \
    git -C /tmp/extensions/FlexGEMM checkout 6dd94a859c26ee8246888502eada3dd8ad85532e && \
    git -C /tmp/extensions/FlexGEMM submodule update --init --recursive && \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation --no-cache-dir

# ---- o-voxel (will be rebuilt from mounted source if needed) ----
COPY o-voxel /tmp/extensions/o-voxel
RUN pip install /tmp/extensions/o-voxel --no-build-isolation --no-cache-dir

# ---- Clean up build artifacts ----
RUN rm -rf /tmp/extensions

# ---- Service-mode (serve) dependencies ----
# Kept as the last pip layer so changing them never invalidates the
# expensive compiled layers above (flash-attn, nvdiffrast, CuMesh, etc.).
RUN pip install --no-cache-dir \
    fastapi==0.138.0 \
    "uvicorn[standard]==0.49.0" \
    python-multipart==0.0.32 \
    websockets==15.0.1 \
    requests==2.32.5

# ---- Environment variables ----
ENV OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ATTN_BACKEND=flash-attn

# ---- Entrypoint script ----
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# ---- Working directory for the mounted project ----
WORKDIR /workspace/TRELLIS.2

# Inference server port (vLLM-style service mode: `serve`)
EXPOSE 8000
# Gradio web UI port (`app`)
EXPOSE 7860

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["demo"]
