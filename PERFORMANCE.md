# TRELLIS.2 inference performance

Measured on an RTX PRO 6000 Blackwell 96 GB with PyTorch 2.7.1/cu128,
FlashAttention 2.7.4, the 4B BF16 checkpoint, and
`assets/example_image/T.png`. Results are the second warm request at 1024
cascade, 12 sampler steps, 2048 texture, and a 100k decimation target.

| Mode | End-to-end | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| Original service | 38.8 s | — | — |
| Optimized service | 36.6 s | 23.92 GiB | 24.54 GiB |

The optimized path is about 6% faster without changing its generation or
post-processing quality settings. The GLB output loads successfully and remains
near the requested 100k-face target.

## Bottlenecks, in priority order

On the optimized high-quality run:

1. Shape sampling: 10.65 s (the high-resolution cascade dominates).
2. UV unwrapping: 7.05 s.
3. Texture sampling model: 5.43 s.
4. Source-mesh BVH: 3.68 s.
5. Remesh/topology: 2.26 s.
6. GLB export: 1.93 s.
7. Texture sampling/finalization: about 3.11 s combined.

Remeshing and source-mesh attribute projection were tested as possible latency
tradeoffs, but disabling either caused an unacceptable mesh-quality regression.
Both therefore remain mandatory.

## Implemented optimizations

- Full-request `torch.inference_mode`, TF32 for remaining FP32 work, and cuDNN
  fixed-shape autotuning.
- Samplers no longer retain all per-step latent tensors or reconstruct unused
  epsilon/x0 tensors. Intermediate-return behavior remains opt-in.
- CUDA timestep and FlashAttention sequence metadata are cached.
- Service progress bars are disabled; structured stage timings remain.
- Texture attributes use one GPU-to-CPU transfer and two OpenCV inpaint calls
  instead of four transfers and four calls.
- The mounted workspace postprocessor is loaded over the image-baked Python
  copy, so service restarts actually pick up source changes.
- HTTP and WebSocket defaults are aligned at 2048/100k. Remeshing and source
  projection are always enabled.

## Quantization result

The checkpoint flow blocks are already BF16. Eager INT8 weight-only linear
kernels were 40–67% slower than BF16 in representative GEMMs. FP8
weight-only was approximately neutral. Dynamic FP8 only helped the MLP
expansion GEMM; applying it selectively to 150 expansion layers reduced peak
allocated/reserved memory by 1.24/1.37 GiB but made the measured end-to-end
request about one second slower and changed generated geometry slightly.

For that reason the quantized implementation and its additional dependency were
not retained; inference remains BF16.
