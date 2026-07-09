"""
Offline INT8 weight-only (MLP) quantizer for TRELLIS.2.

Writes a self-describing quantized copy of the model directory so inference can
load it directly (trellis2.models.from_pretrained auto-detects the per-checkpoint
"quantization" marker and rebuilds the matching module structure). Also drops the
cascade-unused tex_slat_flow_model_512 by default.

What it does per model in pipeline.json:
  * flow DiTs (keys containing "flow")  -> quantize MLP linears to per-channel
    INT8 weight-only (attention stays bf16), save quantized .safetensors + a .json
    carrying {"quantization": {"scheme": "int8_wo", "targets": ["mlp"]}}.
  * decoders (local "ckpts/..." refs)    -> copied verbatim (they run once and are
    sparse-conv dominated; quantizing them is not worth the quality risk).
  * sibling-repo refs (e.g. the sparse_structure_decoder that lives in
    microsoft/TRELLIS-image-large) -> left in place; the loader resolves them.

Run it INSIDE the trellis2 container with the models volume mounted READ-WRITE
(the service normally mounts it :ro), e.g.:

  docker run --rm --gpus all \
    -v trellis2_models:/models \                 # note: rw, not :ro
    -v "$PWD":/workspace/TRELLIS.2 -w /workspace/TRELLIS.2 \
    trellis2:cu128-torch2.7.1 \
    python quantize_offline_int8_mlp.py

Then point the service at it:  TRELLIS2_MODEL_PATH=/models/microsoft/TRELLIS.2-4B-INT8-MLP
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trellis2 import models
from trellis2.quant_int8 import quantize_module_int8_wo

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("offline_quant")

SCHEME = "int8_wo"
TARGETS = ("mlp",)


def _tensor_bytes(sd: dict) -> int:
    return sum(v.numel() * v.element_size() for v in sd.values())


def quantize_flow_checkpoint(src_base: str, dst_base: str, key: str, device: str) -> tuple[int, int]:
    """Load a bf16 flow checkpoint, quantize its MLP to INT8-wo, and save it plus a
    quantization-tagged json. Returns (orig_bytes, new_bytes).

    If no MLP linears match (e.g. the dense sparse_structure_flow_model), the
    checkpoint is copied verbatim with no quantization marker."""
    model = models.from_pretrained(src_base, device=device)
    orig_bytes = _tensor_bytes(model.state_dict())

    n, saved = quantize_module_int8_wo(model, label=key, scheme=SCHEME, targets=TARGETS,
                                   compile_enabled=False)
    if n == 0:
        log.info("  %-32s no MLP linears matched -> copying bf16 unchanged", key)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        copy_checkpoint(src_base, dst_base, key)
        return orig_bytes, orig_bytes

    # rope_phases are deterministically rebuilt by the constructor and are omitted
    # from the published checkpoints; drop them here too for parity / smaller files.
    sd = {k: v.detach().cpu().contiguous()
          for k, v in model.state_dict().items() if not k.endswith("rope_phases")}
    new_bytes = _tensor_bytes(sd)

    os.makedirs(os.path.dirname(dst_base), exist_ok=True)
    save_file(sd, f"{dst_base}.safetensors")

    with open(f"{src_base}.json", "r") as f:
        cfg = json.load(f)
    cfg["quantization"] = {"scheme": SCHEME, "targets": list(TARGETS), "label": key}
    with open(f"{dst_base}.json", "w") as f:
        json.dump(cfg, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("  quantized %-32s %d MLP linears | %.2f -> %.2f GB",
             key, n, orig_bytes / 1e9, new_bytes / 1e9)
    return orig_bytes, new_bytes


def copy_checkpoint(src_base: str, dst_base: str, key: str) -> None:
    os.makedirs(os.path.dirname(dst_base), exist_ok=True)
    shutil.copy2(f"{src_base}.json", f"{dst_base}.json")
    shutil.copy2(f"{src_base}.safetensors", f"{dst_base}.safetensors")
    log.info("  copied    %-32s (unquantized)", key)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/models/microsoft/TRELLIS.2-4B")
    ap.add_argument("--dst", default="/models/microsoft/TRELLIS.2-4B-INT8-MLP")
    ap.add_argument("--drop", nargs="*", default=["tex_slat_flow_model_512"],
                    help="pipeline model keys to omit entirely (default: tex_slat_flow_model_512)")
    ap.add_argument("--keep-bf16", nargs="*", default=["shape_slat_flow_model_512"],
                    help="flow model keys to copy in bf16 instead of quantizing. Default keeps "
                         "shape_slat_flow_model_512 (the cascade coarse driver) exact, since "
                         "quantizing it amplifies geometry error through the 512->1024 cascade.")
    ap.add_argument("--copy-encoder", default="ckpts/shape_enc_next_dc_f16c32_fp16",
                    help="extra checkpoint (the shape encoder) to copy verbatim so the quantized "
                         "directory also serves texture-only (image+mesh) requests. It is not part "
                         "of pipeline.json, so it is copied explicitly. Pass empty to skip.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    src, dst, drop, keep_bf16 = args.src, args.dst, set(args.drop), set(args.keep_bf16)
    log.info("src=%s", src)
    log.info("dst=%s", dst)
    log.info("scheme=%s targets=%s drop=%s keep_bf16=%s device=%s",
             SCHEME, TARGETS, sorted(drop), sorted(keep_bf16), args.device)

    with open(os.path.join(src, "pipeline.json"), "r") as f:
        pipeline_cfg = json.load(f)
    model_refs = pipeline_cfg["args"]["models"]

    os.makedirs(dst, exist_ok=True)
    t0 = time.perf_counter()
    total_orig = total_new = 0
    kept_models: dict[str, str] = {}

    for key, ref in model_refs.items():
        if key in drop:
            log.info("  DROPPED   %-32s (%s)", key, ref)
            continue
        kept_models[key] = ref  # ref unchanged; same relative filename in dst

        if not ref.startswith("ckpts/"):
            # sibling-repo checkpoint (e.g. TRELLIS-image-large) — resolved at load.
            log.info("  external  %-32s -> %s (left in place)", key, ref)
            continue

        src_base = os.path.join(src, ref)
        dst_base = os.path.join(dst, ref)
        if "flow" in key and key not in keep_bf16:
            o, n = quantize_flow_checkpoint(src_base, dst_base, key, args.device)
            total_orig += o
            total_new += n
        else:
            # decoders, and flow models explicitly kept in bf16 (e.g. shape_512)
            copy_checkpoint(src_base, dst_base, key)

    # Write the trimmed pipeline.json (dropped models removed; everything else kept).
    pipeline_cfg["args"]["models"] = kept_models
    with open(os.path.join(dst, "pipeline.json"), "w") as f:
        json.dump(pipeline_cfg, f, indent=2)

    # Texture-only mode (image + mesh -> textured mesh) additionally needs the
    # shape encoder, which is NOT listed in pipeline.json (it belongs to the
    # texturing pipeline). Copy it verbatim (it is fp16 and left unquantized) so
    # this quantized directory can also serve texture-only requests.
    enc_ref = args.copy_encoder
    if enc_ref:
        enc_src = os.path.join(src, enc_ref)
        if os.path.exists(f"{enc_src}.safetensors") and os.path.exists(f"{enc_src}.json"):
            copy_checkpoint(enc_src, os.path.join(dst, enc_ref), "shape_slat_encoder")
        else:
            log.warning("  shape encoder not found at %s(.json/.safetensors); "
                        "texture-only mode will be unavailable against this dir", enc_src)

    log.info("=" * 64)
    log.info("done in %.1f min | flow weights %.2f -> %.2f GB (saved %.2f GB)",
             (time.perf_counter() - t0) / 60, total_orig / 1e9, total_new / 1e9,
             (total_orig - total_new) / 1e9)
    log.info("output: %s", dst)
    log.info("serve with: TRELLIS2_MODEL_PATH=%s  (+ TRELLIS2_OFFLOAD=1 for <=16GB)", dst)
    log.info("note: texturing_pipeline.json is NOT written; this dir targets the image-to-3D "
             "pipeline, plus the shape encoder copied above for texture-only (image+mesh) mode.")


if __name__ == "__main__":
    main()
