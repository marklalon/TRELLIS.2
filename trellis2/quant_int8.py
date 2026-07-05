"""
INT8 weight-only (W8A16) quantization for the TRELLIS.2 flow DiT models.

Motivation
----------
The three 1.3B flow transformers (sparse-structure / shape-SLat / texture-SLat)
dominate both the resident VRAM footprint (~2.6GB each, bf16) and the sampling
latency. Storing their transformer-block linear weights in INT8 halves that
weight memory (the enabler for 16GB cards).

Scheme
------
Per-channel symmetric INT8 weight-only:
  * weights  -> INT8 per-output-channel, computed once at convert time; the bf16
    master weight is dropped so the memory is actually reclaimed.
  * activations -> always bf16 (never quantized), so activation-outlier errors
    that plague W8A8 schemes are completely avoided.
GEMM is plain ``torch.nn.functional.linear`` (dequant → bf16 matmul). The win
is halved weight memory + near-bf16 quality — no FP8 tensor-core speedup.

Tunables (env vars, read by :func:`quantize_module_int8_wo`)
--------------------------------------------------------
  TRELLIS2_FP8_TARGETS    comma list of {mlp, attn}. Default "mlp,attn".
                          "mlp" only skips the small attention projections.
  TRELLIS2_FP8_SKIP_FIRST keep the first N transformer blocks in bf16 (int, 0).
  TRELLIS2_FP8_SKIP_LAST  keep the last N transformer blocks in bf16 (int, 0).
                          Early/late blocks are the most quality-sensitive, so
                          this trades a little memory for lower geometry drift.
  TRELLIS2_FP8_COMPILE    "1" to torch.compile the dequant + GEMM into a single
                          fused kernel (dynamic shapes).

Quality-sensitive / tiny layers (``input_layer``, ``out_layer``,
``adaLN_modulation``, ``t_embedder``) are always kept in bf16.

This module is inert unless explicitly invoked and never changes default behaviour.
"""
from __future__ import annotations

import logging
import os
import re

import torch
import torch.nn as nn

from .modules.sparse.linear import SparseLinear

logger = logging.getLogger("trellis2.quant_int8")

# Attention projection leaves (plain nn.Linear, called on a dense feature tensor).
_ATTN_LEAVES = {"to_qkv", "to_q", "to_kv", "to_out"}
_BLOCK_RE = re.compile(r"blocks\.(\d+)\.")


def _quantize_weight_int8_perchannel(w: torch.Tensor):
    """INT8 per-output-channel symmetric quant. Returns int8 ``[N,K]`` and a
    float32 per-channel scale ``[N]``."""
    scale = (w.detach().abs().amax(dim=1).float() / 127.0).clamp(min=1e-12)  # [N]
    q = (w.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def _int8_wo_apply(x, w_int8, w_scale, bias, out_dtype):
    # Dequant in fp32 (scale is fp32) then cast to the compute dtype; a single
    # fused kernel under torch.compile, otherwise a cheap elementwise + GEMM.
    w = (w_int8.to(torch.float32) * w_scale.unsqueeze(1)).to(out_dtype)
    return torch.nn.functional.linear(x, w, bias)


_COMPILED_INT8_WO = None


def _get_int8_wo_apply(compile_enabled: bool):
    global _COMPILED_INT8_WO
    if not compile_enabled:
        return _int8_wo_apply
    if _COMPILED_INT8_WO is None:
        _COMPILED_INT8_WO = torch.compile(_int8_wo_apply, dynamic=True)
    return _COMPILED_INT8_WO


class Int8WeightOnlyLinear(nn.Module):
    """W8A16 replacement for ``nn.Linear`` on a dense ``[*, in]`` tensor."""

    def __init__(self, linear: nn.Linear, apply_fn=_int8_wo_apply):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self._apply_fn = apply_fn
        weight_int8, weight_scale = _quantize_weight_int8_perchannel(linear.weight.data)
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("weight_scale", weight_scale)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype if x.is_floating_point() else torch.bfloat16
        bias = self.bias.to(out_dtype) if self.bias is not None else None
        return self._apply_fn(x, self.weight_int8, self.weight_scale, bias, out_dtype)


class Int8WeightOnlySparseLinear(nn.Module):
    """W8A16 replacement for the MLP ``SparseLinear`` (``VarLenTensor`` in/out)."""

    def __init__(self, linear: SparseLinear, apply_fn=_int8_wo_apply):
        super().__init__()
        self.linear = Int8WeightOnlyLinear(linear, apply_fn=apply_fn)

    def forward(self, x):  # x: VarLenTensor
        return x.replace(self.linear(x.feats))


# scheme -> (dense_class, sparse_class, apply_getter, requires_dim16)
_SCHEMES = {
    "int8_wo": (Int8WeightOnlyLinear, Int8WeightOnlySparseLinear, _get_int8_wo_apply, False),
}


def _block_index(mod_name: str):
    m = _BLOCK_RE.search(mod_name)
    return int(m.group(1)) if m else None


def quantize_module_int8_wo(
    model: nn.Module,
    label: str = "model",
    *,
    scheme: str = "int8_wo",
    targets=("mlp", "attn"),
    skip_first: int = 0,
    skip_last: int = 0,
    compile_enabled: bool = False,
) -> tuple[int, int]:
    """In-place 8-bit conversion of the eligible transformer-block linears of one
    flow model. Returns ``(num_layers_converted, bytes_saved)``.

    ``scheme`` selects the quantization: ``"int8_wo"`` (per-channel INT8
    weight-only, bf16 activations).

    The original bf16 ``nn.Linear`` objects are dereferenced by the ``setattr``
    swap; call ``torch.cuda.empty_cache()`` afterwards to reclaim the VRAM.
    """
    targets = set(targets)
    dense_cls, sparse_cls, apply_getter, requires_dim16 = _SCHEMES[scheme]
    apply_fn = apply_getter(compile_enabled)
    num_blocks = len(model.blocks) if hasattr(model, "blocks") else None

    converted = 0
    bytes_saved = 0
    skipped = 0
    for mod_name, module in list(model.named_modules()):
        leaf = mod_name.rsplit(".", 1)[-1]
        is_sparse = isinstance(module, SparseLinear)
        is_plain = isinstance(module, nn.Linear) and not is_sparse

        if is_sparse and ".mlp.mlp." in mod_name:
            kind = "mlp"
        elif is_plain and leaf in _ATTN_LEAVES:
            kind = "attn"
        else:
            continue
        if kind not in targets:
            continue

        # Keep the most quality-sensitive first/last blocks in bf16.
        if num_blocks is not None and (skip_first or skip_last):
            idx = _block_index(mod_name)
            if idx is not None and (idx < skip_first or idx >= num_blocks - skip_last):
                continue

        # ``requires_dim16`` is ``False`` for the only active scheme (int8_wo),
        # so the dim-alignment check below is a no-op — kept as future-proofing
        # in case a dim-sensitive scheme is added later.
        if requires_dim16 and (module.in_features % 16 or module.out_features % 16):
            skipped += 1
            continue

        parent = model.get_submodule(mod_name.rsplit(".", 1)[0]) if "." in mod_name else model
        weight_bytes = module.weight.numel() * module.weight.element_size()
        make = sparse_cls if kind == "mlp" else dense_cls
        setattr(parent, leaf, make(module, apply_fn=apply_fn))
        converted += 1
        bytes_saved += weight_bytes // 2   # 8-bit is half of bf16/fp16

    logger.info(
        "FP8 [%s]: converted %d linear(s), skipped %d (misaligned), ~%.2f GB saved",
        label, converted, skipped, bytes_saved / 1e9,
    )
    return converted, bytes_saved


def apply_quantization_from_config(model: nn.Module, quant_cfg: dict, *, compile_enabled=None):
    """Rebuild the quantized module structure described by a checkpoint's
    ``quantization`` config, so an offline-quantized state_dict loads cleanly.

    Offline export and inference call the *same* swap, guaranteeing the module tree
    (and therefore the state_dict keys) match. The freshly-constructed weights the
    swap quantizes here are placeholders — ``load_state_dict(assign=True)`` replaces
    the int8/scale buffers with the saved ones immediately afterwards.
    """
    scheme = quant_cfg.get("scheme", "int8_wo")
    targets = tuple(quant_cfg.get("targets", ("mlp", "attn")))
    skip_first = int(quant_cfg.get("skip_first", 0))
    skip_last = int(quant_cfg.get("skip_last", 0))
    if compile_enabled is None:
        compile_enabled = os.environ.get("TRELLIS2_FP8_COMPILE", "0") == "1"
    return quantize_module_int8_wo(
        model, label=quant_cfg.get("label", "checkpoint"), scheme=scheme,
        targets=targets, skip_first=skip_first, skip_last=skip_last,
        compile_enabled=compile_enabled,
    )

