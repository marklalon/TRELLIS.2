import importlib
import contextlib
import logging
import threading

logger = logging.getLogger("trellis2.models")

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# Reentrant, thread-safe guard that turns the random weight initializers into
# no-ops. Constructing a multi-billion parameter model runs full random init
# (nn.Linear.reset_parameters, plus the explicit xavier_/normal_ passes) on
# every parameter on the CPU -- which dominates load time and is pure waste
# because load_state_dict overwrites all of it. We skip only the random fills;
# cheap deterministic inits (constant_/zeros_/ones_) are left intact so any
# parameter that happens to be missing from the checkpoint keeps a sane value.
_INIT_SKIP_NAMES = (
    "kaiming_uniform_", "kaiming_normal_",
    "xavier_uniform_", "xavier_normal_",
    "uniform_", "normal_", "trunc_normal_",
)
_init_skip_lock = threading.Lock()
_init_skip_depth = 0
_init_saved = {}


@contextlib.contextmanager
def skip_random_init():
    """No-op the RNG-based weight initializers for the duration of the block."""
    global _init_skip_depth
    import torch.nn.init as init

    def _noop(tensor, *args, **kwargs):
        return tensor

    with _init_skip_lock:
        if _init_skip_depth == 0:
            for name in _INIT_SKIP_NAMES:
                if hasattr(init, name):
                    _init_saved[name] = getattr(init, name)
                    setattr(init, name, _noop)
        _init_skip_depth += 1
    try:
        yield
    finally:
        with _init_skip_lock:
            _init_skip_depth -= 1
            if _init_skip_depth == 0:
                for name, fn in _init_saved.items():
                    setattr(init, name, fn)
                _init_saved.clear()


def from_pretrained(path: str, device=None, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        device: Optional device to load the weights directly onto (e.g. ``"cuda"``).
                When given, weights are read straight to that device and assigned
                in place, avoiding a separate CPU->GPU transfer. Defaults to CPU.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file

    is_local_path = path.startswith('/') or '\\' in path

    if is_local_path:
        cfg = f"{path}.json"
        wts = f"{path}.safetensors"
        if not (os.path.exists(cfg) and os.path.exists(wts)):
            raise FileNotFoundError(
                f"Local model checkpoint not found.\n"
                f"  Expected: {cfg}\n"
                f"  Expected: {wts}\n"
                f"Ensure the models Docker volume is mounted."
            )
        config_file = cfg
        model_file = wts
    else:
        from huggingface_hub import hf_hub_download
        clean_path = path.replace('\\', '/').strip('/')
        path_parts = clean_path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:]) if len(path_parts) > 2 else path_parts[-1]
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    # Skip the wasted random initialization; weights come from the checkpoint.
    with skip_random_init():
        model = __getattr__(config['name'])(**config['args'], **kwargs)

    load_device = str(device) if device is not None else "cpu"
    state_dict = load_file(model_file, device=load_device)
    if device is not None:
        # Load straight to the GPU and assign the tensors to the parameters
        # (no extra copy / no second transfer). The model defines the intended
        # per-module dtypes (e.g. bf16 blocks, fp32 elsewhere via convert_to),
        # so cast each checkpoint tensor to the matching param/buffer dtype
        # before assigning -- otherwise inference hits dtype mismatches.
        target_dtypes = {k: v.dtype for k, v in model.state_dict().items()}
        for k, v in list(state_dict.items()):
            td = target_dtypes.get(k)
            if td is not None and v.dtype != td and v.is_floating_point():
                state_dict[k] = v.to(td)
        result = model.load_state_dict(state_dict, strict=False, assign=True)
        # Move anything not covered by the checkpoint (computed buffers, and any
        # skip-init params absent from the state dict) onto the target device.
        model.to(device)
    else:
        # CPU path: in-place load casts the checkpoint to each param's dtype.
        result = model.load_state_dict(state_dict, strict=False)
    # RoPE phases are deterministically rebuilt by the model constructor and
    # are intentionally absent from some published checkpoints.
    missing_keys = [
        key for key in getattr(result, "missing_keys", [])
        if key != "rope_phases"
    ]
    if missing_keys:
        logger.warning(
            "Checkpoint '%s' is missing %d key(s); those params keep uninitialized "
            "values (e.g. %s)", path, len(missing_keys), missing_keys[:5]
        )

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
