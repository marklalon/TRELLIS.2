# TRELLIS.2 Docker Image (trellis2:cu128-torch2.7.1)
# Base: base-cuda:cu128-torch2.7.1 (Ubuntu 24.04 + Python 3.11-dev + CUDA 12.8 +
#       cuda-toolkit-12-8 + build-essential + PyTorch 2.7.1 + flash-attn 2.7.4.post1).
#       See D:\AI\llm-runtime\base-cuda.
#
# Single-stage on purpose: the base already ships the full compiler toolchain
# (nvcc + gcc + Python headers), so there is nothing to apt-install for building.
# A multi-stage split would NOT shrink the image either — the toolchain lives in
# the shared base layer that any final stage inherits — and it must stay anyway:
# FlexGEMM autotune and Triton JIT-compile kernels at *runtime* (see
# TRITON_CACHE_DIR / FLEX_GEMM_AUTOTUNE_CACHE_PATH in docker-compose), which
# needs ptxas and a host compiler present in the running container.

FROM base-cuda:cu128-torch2.7.1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST="12.0" \
    MAX_JOBS=4

# ---- Runtime shared libraries the base does not carry ----
# The compiler toolchain, git and curl are already in the base. Pillow/
# opencv-headless wheels bundle their own image codecs, so no system libjpeg/
# libwebp is needed. The GL/X11 libs back nvdiffrast's GL path. libgl1 replaces
# libgl1-mesa-glx, which was removed in Ubuntu 24.04.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0t64 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Upgrade pip ----
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

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

# Pillow: the stock Pillow pulled in by torchvision (~12.x) is kept as-is. Its
# manylinux wheel already bundles libjpeg-turbo/libwebp/zlib, so WebP export and
# JPEG/PNG decoding work with no system image codecs and no pillow-simd rebuild.

# ---- nvdiffrast ----
RUN mkdir -p /tmp/extensions && \
    git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast && \
    pip install /tmp/extensions/nvdiffrast --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/extensions/nvdiffrast

# ---- nvdiffrec (renderutils branch, pinned commit) ----
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec && \
    git -C /tmp/extensions/nvdiffrec checkout b296927cc7fd01c2ac1087c8065c4d7248f72da4 && \
    pip install /tmp/extensions/nvdiffrec --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/extensions/nvdiffrec

# ---- CuMesh (pinned commit) ----
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh && \
    git -C /tmp/extensions/CuMesh checkout 12289e1062f0603f2f0d0771b02e1395d247f26f && \
    git -C /tmp/extensions/CuMesh submodule update --init --recursive && \
    pip install /tmp/extensions/CuMesh --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/extensions/CuMesh

# ---- Patch CuMesh remeshing.py for torch.cross deprecated API ----
RUN sed -i '/torch\.cross(/{/dim=/!s/)$/, dim=-1)/;}' \
        /usr/local/lib/python3.11/dist-packages/cumesh/remeshing.py

# ---- Rebuild xatlas from patched fork (9x faster UV packing) ----
# Upstream xatlas PackCharts submits one task scheduler task per chart; on
# meshes with many small charts the submit/notify overhead dominates. The
# fork chunks charts into ~threadCount*4 tasks with bit-identical output.
ARG CUMESH_XATLAS_REF=dca43c690452d3a0ef8362814085e7f26375364b
RUN set -eux; \
    raw="https://raw.githubusercontent.com/marklalon/CuMesh/${CUMESH_XATLAS_REF}/third_party/xatlas"; \
    mkdir -p /tmp/xatlas && cd /tmp/xatlas; \
    curl -fsSL "$raw/xatlas.cpp"  -o xatlas.cpp; \
    curl -fsSL "$raw/xatlas.h"    -o xatlas.h; \
    curl -fsSL "$raw/binding.cpp" -o binding.cpp; \
    printf '%s\n' \
      'from setuptools import setup' \
      'from torch.utils.cpp_extension import CppExtension, BuildExtension' \
      'setup(ext_modules=[CppExtension("_cumesh_xatlas", ["xatlas.cpp", "binding.cpp"],' \
      '      extra_compile_args=["-O3", "-std=c++17"])], cmdclass={"build_ext": BuildExtension})' \
      > setup_xatlas.py; \
    python setup_xatlas.py build_ext --inplace; \
    cp _cumesh_xatlas*.so /usr/local/lib/python3.11/dist-packages/cumesh/; \
    python -c "import torch, importlib.util, glob; \
p = glob.glob('/usr/local/lib/python3.11/dist-packages/cumesh/_cumesh_xatlas*.so')[0]; \
s = importlib.util.spec_from_file_location('_cumesh_xatlas', p); m = importlib.util.module_from_spec(s); \
s.loader.exec_module(m); m.Atlas(); print('optimized _cumesh_xatlas loads OK:', p)"; \
    cd / && rm -rf /tmp/xatlas

# ---- FlexGEMM (pinned commit) ----
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM && \
    git -C /tmp/extensions/FlexGEMM checkout 6dd94a859c26ee8246888502eada3dd8ad85532e && \
    git -C /tmp/extensions/FlexGEMM submodule update --init --recursive && \
    pip install /tmp/extensions/FlexGEMM --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/extensions/FlexGEMM

# ---- Service-mode (serve) dependencies ----
# Kept as the last pip layer so changing them never invalidates the expensive
# compiled layers above.
RUN pip install --no-cache-dir \
    fastapi==0.138.0 \
    "uvicorn[standard]==0.49.0" \
    python-multipart==0.0.32 \
    websockets==15.0.1 \
    requests==2.32.5

# ---- o-voxel (local source) ----
# Last so editing local o-voxel source only invalidates this layer onward.
COPY o-voxel /tmp/extensions/o-voxel
RUN pip install /tmp/extensions/o-voxel --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/extensions/o-voxel

# ---- SageAttention (pinned commit) ----
# INT8-QK / FP8-PV fused attention. On sm120 (Blackwell workstation/consumer)
# `sageattn` dispatches to sageattn_qk_int8_pv_fp8_cuda (per-warp, fp32+fp16
# accum). Compiled only for sm_120 to keep build time bounded; needs CUDA >=12.8
# for compute capability 12.0 (satisfied by the cu128 base). Selected at runtime
# via ATTN_BACKEND=sage. Placed after o-voxel so editing local o-voxel source
# does not trigger a SageAttention recompile.
ARG SAGEATTENTION_REF=d1a57a5
RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/extensions/SageAttention && \
    git -C /tmp/extensions/SageAttention checkout ${SAGEATTENTION_REF} && \
    TORCH_CUDA_ARCH_LIST="12.0" EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 4" MAX_JOBS=4 \
        pip install /tmp/extensions/SageAttention --no-build-isolation --no-cache-dir && \
    rm -rf /tmp/extensions/SageAttention

# ---- Trim build cruft from the newly added packages ----
RUN find /usr/local/lib/python3.11/dist-packages -depth -type d -name '__pycache__' -exec rm -rf {} + ; \
    find /usr/local/lib/python3.11/dist-packages -type f -name '*.pyc' -delete ; \
    true

# ---- Environment variables ----
ENV OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ATTN_BACKEND=cudnn_sdpa

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
