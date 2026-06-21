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
    imageio \
    imageio-ffmpeg \
    tqdm \
    easydict \
    opencv-python-headless \
    ninja \
    trimesh \
    transformers \
    gradio==6.0.1 \
    tensorboard \
    pandas \
    lpips \
    zstandard \
    huggingface_hub \
    safetensors \
    plyfile \
    kornia \
    timm \
    scipy \
    pyyaml

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

# ---- nvdiffrec (renderutils branch) ----
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    pip install /tmp/extensions/nvdiffrec --no-build-isolation --no-cache-dir

# ---- CuMesh ----
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive && \
    pip install /tmp/extensions/CuMesh --no-build-isolation --no-cache-dir

# ---- FlexGEMM ----
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive && \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation --no-cache-dir

# ---- o-voxel (will be rebuilt from mounted source if needed) ----
COPY o-voxel /tmp/extensions/o-voxel
RUN pip install /tmp/extensions/o-voxel --no-build-isolation --no-cache-dir

# ---- Clean up build artifacts ----
RUN rm -rf /tmp/extensions

# ---- Environment variables ----
ENV OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ATTN_BACKEND=flash-attn

# ---- Entrypoint script ----
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# ---- Working directory for the mounted project ----
WORKDIR /workspace/TRELLIS.2

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["demo"]
