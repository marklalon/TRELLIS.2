from typing import *
import torch
import torch.nn as nn
from .. import models


def _load_model_from_local(base_path: str, model_ref: str) -> nn.Module:
    """
    Resolve a model reference from pipeline.json against a local base path.

    model_ref can be:
      - Relative: ``ckpts/ss_flow_img_dit_1_3B_64_bf16``
      - HuggingFace-style: ``microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16``

    When loading locally, HF namespace prefixes are stripped so the checkpoint
    is searched under the local ``ckpts/`` directory.
    """
    import os

    # Candidate 1: direct join
    candidate = f"{base_path}/{model_ref}"
    if os.path.exists(f"{candidate}.json") and os.path.exists(f"{candidate}.safetensors"):
        return models.from_pretrained(candidate)

    # Candidate 2: strip HF org/repo prefix (e.g. microsoft/TRELLIS-image-large)
    parts = model_ref.split('/')
    if len(parts) >= 3:
        stripped = '/'.join(parts[2:])
        candidate = f"{base_path}/{stripped}"
        if os.path.exists(f"{candidate}.json") and os.path.exists(f"{candidate}.safetensors"):
            return models.from_pretrained(candidate)

    # Candidate 3: check basename under base_path/ckpts/
    basename = parts[-1]
    candidate = f"{base_path}/ckpts/{basename}"
    if os.path.exists(f"{candidate}.json") and os.path.exists(f"{candidate}.safetensors"):
        return models.from_pretrained(candidate)

    # Candidate 4: sibling HF repo under the same volume root
    # e.g. microsoft/TRELLIS-image-large/ckpts/ss_dec_...  ->
    #      /models/microsoft/TRELLIS-image-large/ckpts/ss_dec_...
    if len(parts) >= 3:
        org = parts[0]
        repo = parts[1]
        sub = '/'.join(parts[2:])
        # base_path is like /models/org/current_repo; sibling is /models/org/other_repo
        base_parts = base_path.rstrip('/').split('/')
        if len(base_parts) >= 3:
            # /models/org/current_repo -> /models/org
            volume_root = '/'.join(base_parts[:-1])
            candidate = f"{volume_root}/{repo}/{sub}"
            if os.path.exists(f"{candidate}.json") and os.path.exists(f"{candidate}.safetensors"):
                return models.from_pretrained(candidate)

    raise FileNotFoundError(
        f"Local checkpoint not found for model reference '{model_ref}'.\n"
        f"  Base path: {base_path}\n"
        f"  Tried: 1) {base_path}/{model_ref}\n"
        f"         2) strip HF prefix -> {base_path}/{stripped if len(parts) >= 3 else 'N/A'}\n"
        f"         3) basename under ckpts/ -> {base_path}/ckpts/{basename}\n"
        f"         4) sibling repo path\n"
        f"Ensure all model checkpoints are present in the volumes."
    )


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pipeline":
        """
        Load a pretrained model.

        If path is a local filesystem path (absolute or contains backslashes), models are
        loaded exclusively from disk. Otherwise, they are downloaded from HuggingFace Hub.
        """
        import os
        import json

        is_local_path = os.path.isabs(path) or '\\' in path

        if is_local_path:
            local_cfg = f"{path}/{config_file}"
            if not os.path.exists(local_cfg):
                raise FileNotFoundError(
                    f"Local model path not found: {local_cfg}\n"
                    f"Ensure the models Docker volume is mounted and contains pipeline.json."
                )
            config_file = local_cfg
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            if hasattr(cls, 'model_names_to_load') and k not in cls.model_names_to_load:
                continue
            if is_local_path:
                _models[k] = _load_model_from_local(path, v)
            else:
                _models[k] = models.from_pretrained(f"{path}/{v}")

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        new_pipeline._is_local_path = is_local_path
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))