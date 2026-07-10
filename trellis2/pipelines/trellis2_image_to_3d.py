from typing import *
import os
import logging
from contextlib import contextmanager
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


class ActiveTokenLimitExceeded(RuntimeError):
    """Raised when the sparse structure decodes to more active voxels (tokens)
    than the configured ceiling. Peak VRAM scales ~linearly with this count, so
    it is the earliest reliable predictor of an over-budget request -- known at
    ~0.15s in, before the expensive shape/texture sampling and decode. The
    server maps this to a clean client error instead of risking an OOM that
    would take down concurrent requests.
    """

    def __init__(self, num_tokens: int, limit: int):
        self.num_tokens = num_tokens
        self.limit = limit
        super().__init__(
            f"model too complex: {num_tokens} active tokens exceeds limit "
            f"{limit}; retry with a simpler/less-filled image or 'mesh' mode"
        )


logger = logging.getLogger("trellis2.pipeline")
# serve.py configures logging only on its own "trellis2.serve" logger and never
# touches the root logger, so this logger would otherwise inherit root's WARNING
# level with no handler and drop every INFO line. Make it self-sufficient.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _log_decode_peak(tag: str) -> None:
    """Attribute the decode-stage activation peak to a sub-op.

    ``max_memory_allocated`` captures the torch-side transient peak of a UNet
    forward even though ``no_grad`` frees it immediately -- which boundary and
    background probes both miss. Gated by the shared OVOXEL_VRAM_LOG switch.
    """
    if os.environ.get("OVOXEL_VRAM_LOG", "1") == "0" or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    logger.info("[decode] %-12s peak_alloc=%.2fG now_alloc=%.2fG", tag,
                torch.cuda.max_memory_allocated() / 1e9,
                torch.cuda.memory_allocated() / 1e9)
    torch.cuda.reset_peak_memory_stats()


def _resolve_model_names_to_local(args: dict) -> None:
    """
    Replace HF model names (e.g. ``facebook/dinov3-vitl16-pretrain-lvd1689m``)
    with local filesystem paths under ``/models/`` when loading from disk.
    """
    for section, key in [
        ('image_cond_model', 'model_name'),
        ('rembg_model', 'model_name'),
    ]:
        sec = args.get(section, {})
        name = sec.get('args', {}).get(key, '')
        if name and '/' in name and not os.path.isabs(name):
            local_path = f"/models/{name}"
            if os.path.exists(local_path):
                sec['args'][key] = local_path


class Trellis2ImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis2 image-to-3D models.

    .. note::
        The ``'512'`` pipeline type reuses ``tex_slat_flow_model_1024`` for
        texturing (the dedicated 512 texture model was dropped). The 1024
        architecture is compatible because its RoPE position encoding handles
        the 512 shape's smaller coordinate range, and fewer tokens keep
        texture-stage VRAM below the 1024 path.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model (Callable): The image conditioning model.
        rembg_model (Callable): The model for removing background.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model: Callable = None,
        rembg_model: Callable = None,
        default_pipeline_type: str = '512',
    ):
        # CPU offload of the idle 1.3B flow models, which are dead weight during
        # the VRAM peak (the VAE decoders). TRELLIS2_OFFLOAD selects the policy:
        #   "0"   (default) - no offload, flow models always resident.
        #   "1"/"stage"     - page each flow model in for its own sampling stage
        #                     and out immediately after. Lowest resident VRAM but
        #                     stalls every stage (6 transfers/request).
        #   "decode"        - keep flow models resident through all sampling
        #                     stages (fast, no per-stage stalls), then evict them
        #                     once before decode so decode runs without them.
        #                     The server prewarms them back onto the GPU *after*
        #                     the request finishes (see prewarm_flow_models),
        #                     while the GPU is idle, so the next request's
        #                     sampling starts warm without stalling on a CPU->GPU
        #                     transfer -- and without holding up any request,
        #                     since the prewarm is skipped whenever another
        #                     request is already sampling. The next request still
        #                     evicts them before its own decode, so the decode
        #                     VRAM peak is unchanged.
        _mode = os.environ.get("TRELLIS2_OFFLOAD", "0").strip().lower()
        if _mode in ("1", "true", "stage"):
            self._offload_mode = "stage"
        elif _mode == "decode":
            self._offload_mode = "decode"
        else:
            self._offload_mode = None
        self._offload = self._offload_mode is not None
        # Flow-model keys the last request evicted before decode; "decode" mode
        # prewarms exactly these back onto the GPU after the request finishes.
        self._pending_prewarm: List[str] = []
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model = image_cond_model
        self.rembg_model = rembg_model
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_file: str = "pipeline.json",
        progress_callback: Optional[Callable[[str], None]] = None,
        device=None,
        rembg_model_path: Optional[str] = None,
    ) -> "Trellis2ImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
            device: Optional device to load checkpoint weights directly onto
                    (e.g. ``"cuda"``). Defaults to CPU.
            rembg_model_path: Optional local path or Hugging Face repository
                              overriding the background-removal model in the
                              pipeline configuration.
        """
        pipeline = super().from_pretrained(path, config_file, progress_callback, device=device)
        args = pipeline._pretrained_args
        is_local_path = getattr(pipeline, '_is_local_path', False)

        if rembg_model_path is not None:
            args['rembg_model']['args']['model_name'] = rembg_model_path

        # Resolve HF model names to local paths when loading from disk
        if is_local_path:
            _resolve_model_names_to_local(args)

        if progress_callback is not None:
            progress_callback("initializing samplers")
        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        if progress_callback is not None:
            progress_callback("loading image encoder")
        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        if progress_callback is not None:
            progress_callback("loading background-removal model")
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])
        
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '512')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        super().to(device)
        self.image_cond_model.to(device)
        if self.rembg_model is not None:
            self.rembg_model.to(device)

    @staticmethod
    def _is_offloadable(model_key: str) -> bool:
        """Models used in a single early stage that then sit idle through the
        peak-VRAM decode, so they are the ones we page to CPU: the big flow DiTs
        (one sampling stage each) plus the sparse-structure decoder (runs only
        during sparse-structure sampling, before shape/texture sampling)."""
        return 'flow' in model_key or model_key == 'sparse_structure_decoder'

    def cuda(self) -> None:
        super().cuda()
        if self._offload_mode == "stage":
            # In "stage" mode, start flow models on CPU; _resident pages them
            # in and out each stage for minimum VRAM.
            offloaded = []
            for key, model in self.models.items():
                if self._is_offloadable(key) and isinstance(model, nn.Module):
                    model.cpu()
                    offloaded.append(key)
            torch.cuda.empty_cache()
            logger.info(
                "CPU offload=%s: flow models start on CPU (%s)",
                self._offload_mode, ", ".join(offloaded),
            )

    @contextmanager
    def _resident(self, model):
        """Ensure a flow model is on the GPU for its sampling stage.

        In "stage" mode it is evicted back to the CPU immediately after (lowest
        VRAM, but a transfer every stage). In "decode" mode it is left resident
        so sampling never stalls; the bulk eviction happens once in
        ``_evict_flow_models`` before decode. A no-op unless offload is enabled.
        """
        if not self._offload or not isinstance(model, nn.Module):
            yield model
            return
        model.to(self._device)
        try:
            yield model
        finally:
            if self._offload_mode == "stage":
                model.cpu()
                torch.cuda.empty_cache()

    def _evict_flow_models(self) -> None:
        """Page any resident flow model back to the CPU. Used before decode in
        "decode" mode; a no-op in "stage" mode (already evicted) and when off.

        Records the moved keys (the set this request used) in
        ``self._pending_prewarm`` so ``prewarm_flow_models`` can page exactly
        those back onto the GPU once the request has finished.
        """
        if not self._offload:
            return
        moved = []
        for key, model in self.models.items():
            if (self._is_offloadable(key) and isinstance(model, nn.Module)
                    and next(model.parameters()).device.type != 'cpu'):
                model.cpu()
                moved.append(key)
        if moved:
            self._pending_prewarm = moved
            torch.cuda.empty_cache()
            logger.info("evicted idle models to CPU before decode (%s)",
                        ", ".join(moved))

    def prewarm_flow_models(self) -> None:
        """Page the flow models used by the last request back onto the GPU so the
        next request's sampling starts warm instead of stalling on a CPU->GPU
        transfer ("decode" mode only; a no-op otherwise).

        Intended to be called by the server *after* a request finishes, while the
        GPU is idle, so it stays off every request's critical path. The caller
        MUST serialize this against sampling -- i.e. hold the same lock that
        guards ``run`` -- because every flow-model device movement must be
        mutually exclusive. The next request still evicts these before its own
        decode, so prewarming does not raise the decode-stage VRAM peak.
        """
        if not self._offload or self._offload_mode != "decode":
            return
        moved = []
        for key in self._pending_prewarm:
            model = self.models.get(key)
            if (isinstance(model, nn.Module)
                    and next(model.parameters()).device.type == 'cpu'):
                model.to(self._device)
                moved.append(key)
        if moved:
            logger.info("prewarmed flow models to GPU after request (%s)",
                        ", ".join(moved))

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            output = self.rembg_model(input)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        self.image_cond_model.image_size = resolution
        cond = self.image_cond_model(image)
        if not include_neg_cond:
            return {'cond': cond}
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
            step_callback (callable): Optional ``(completed_steps, total_steps)`` callback
                fired after each sampling step, for progress reporting.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        with self._resident(flow_model):
            z_s = self.sparse_structure_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=False,
                tqdm_desc="Sampling sparse structure",
                step_callback=step_callback,
            ).samples

        # Decode sparse structure latent. Wrapped in ``_resident`` so that when it
        # is offloaded (it now counts as offloadable) stage mode pages it onto the
        # GPU only for this forward and back off afterwards, and so it survives the
        # CPU-start that ``cuda()`` applies to offloadable models in stage mode.
        decoder = self.models['sparse_structure_decoder']
        with self._resident(decoder):
            decoded = decoder(z_s)>0
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            step_callback (callable): Optional ``(completed_steps, total_steps)`` callback
                fired after each sampling step, for progress reporting.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        with self._resident(flow_model):
            slat = self.shape_slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=False,
                tqdm_desc="Sampling shape SLat",
                step_callback=step_callback,
            ).samples

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat

    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            step_callback (callable): Optional ``(completed_steps, total_steps)`` callback
                fired after each sampling step, for progress reporting. The callback
                receives (completed, total) spanning both LR and HR sampling phases.
        """
        # LR sampling — the first of two cascade phases (LR → HR). Wrap
        # step_callback so that LR reports completed/total*2 and HR reports
        # (total+completed)/(total*2), giving continuous 0→1 progress across
        # both phases regardless of per-phase step counts.
        def _lr_cb(c: int, t: int) -> None:
            if step_callback is not None:
                step_callback(c, t * 2)
        def _hr_cb(c: int, t: int) -> None:
            if step_callback is not None:
                step_callback(t + c, t * 2)

        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        with self._resident(flow_model_lr):
            slat = self.shape_slat_sampler.sample(
                flow_model_lr,
                noise,
                **lr_cond,
                **sampler_params,
                verbose=False,
                tqdm_desc="Sampling shape SLat",
                step_callback=_lr_cb,
            ).samples
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        # Upsample
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        with self._resident(flow_model):
            slat = self.shape_slat_sampler.sample(
                flow_model,
                noise,
                **cond,
                **sampler_params,
                verbose=False,
                tqdm_desc="Sampling shape SLat",
                step_callback=_hr_cb,
            ).samples

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
        step_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape
            sampler_params (dict): Additional parameters for the sampler.
            step_callback (callable): Optional ``(completed_steps, total_steps)`` callback
                fired after each sampling step, for progress reporting.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        with self._resident(flow_model):
            slat = self.tex_slat_sampler.sample(
                flow_model,
                noise,
                concat_cond=shape_slat,
                **cond,
                **sampler_params,
                verbose=False,
                tqdm_desc="Sampling texture SLat",
                step_callback=step_callback,
            ).samples

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
            progress_callback (callable): Optional ``(percent, stage)`` callback on a
                0..100 scale for sub-stage progress within decode.
        """
        def _report(pct: int, stage: str) -> None:
            if progress_callback is not None:
                progress_callback(pct, stage)

        _log_decode_peak("before")
        _report(0, "decoding shape latent")
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        _log_decode_peak("shape_slat")
        # Mesh-only path: no texture latent was sampled, so return geometry with
        # empty voxel attributes (postprocess exports a white, UV-less mesh).
        if tex_slat is None:
            out_mesh = []
            for i, m in enumerate(meshes):
                m.fill_holes()
                _report(50 + round((i + 1) / len(meshes) * 50), "finalizing meshes")
                out_mesh.append(
                    MeshWithVoxel(
                        m.vertices, m.faces,
                        origin = [-0.5, -0.5, -0.5],
                        voxel_size = 1 / resolution,
                        coords = None,
                        attrs = None,
                        voxel_shape = None,
                        layout=self.pbr_attr_layout
                    )
                )
            _report(100, "decode complete")
            return out_mesh
        _report(30, "decoding texture latent")
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        _log_decode_peak("tex_slat")
        out_mesh = []
        for i, (m, v) in enumerate(zip(meshes, tex_voxels)):
            m.fill_holes()
            _report(50 + round((i + 1) / len(meshes) * 50), "finalizing meshes")
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        max_active_tokens: Optional[int] = None,
        generate_texture: bool = True,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> List[MeshWithVoxel]:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '512', '1024', '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
            max_active_tokens (int): Optional ceiling on the number of active
                voxels decoded from the sparse structure. Peak VRAM scales
                ~linearly with this count, so exceeding it raises
                ``ActiveTokenLimitExceeded`` *before* the expensive
                shape/texture sampling and decode stages -- a service-side guard
                against a single heavy request OOM-ing the process. ``None``
                disables the check.
            generate_texture (bool): Whether to sample/decode the texture latent.
                When False, only the shape is produced (a white, UV-less mesh),
                skipping the texture flow model and texture decoder entirely.
            progress_callback (callable): Optional ``(percent, stage)`` callback.
        """
        def report(percent: int, stage: str) -> None:
            if progress_callback is not None:
                progress_callback(percent, stage)

        report(0, "starting pipeline")
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '512':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            # The 512 texture model was dropped (blurry, VRAM-heavy). The 512
            # pipeline reuses the 1024 texture model: same architecture, RoPE PE
            # handles the 512 shape's smaller coord range, and fewer tokens keep
            # texture-stage VRAM below the 1024 path.
            assert 'tex_slat_flow_model_1024' in self.models, "No texture SLat flow model found (512 pipeline reuses the 1024 texture model)."
        elif pipeline_type == '1024':
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
        else:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")
        
        if preprocess_image:
            report(5, "preprocessing image")
            image = self.preprocess_image(image)
        report(10, "encoding image conditions")
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        # cond_1024 is needed by every pipeline type: the non-512 types use it for
        # shape/texture, and the 512 type now uses it for the 1024 texture model.
        cond_1024 = self.get_cond([image], 1024)
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        report(20, "sampling sparse structure")
        coords = self.sample_sparse_structure(
            cond_512, ss_res,
            num_samples, sparse_structure_sampler_params,
            step_callback=lambda c, t: report(20 + round(c / max(t, 1) * 20), "sampling sparse structure"),
        )
        # VRAM guard: bail out here, before the two big sampling stages and the
        # peak-VRAM decode, if the object decoded to more active voxels than the
        # service budgets for. This is the earliest point the request's peak is
        # predictable (peak ~ number of active tokens).
        num_active_tokens = int(coords.shape[0])
        if max_active_tokens is not None and num_active_tokens > max_active_tokens:
            logger.warning(
                "rejecting request: %d active tokens exceeds limit %d",
                num_active_tokens, max_active_tokens,
            )
            raise ActiveTokenLimitExceeded(num_active_tokens, max_active_tokens)
        logger.info("active tokens=%d (limit=%s)", num_active_tokens,
                    max_active_tokens if max_active_tokens is not None else "off")
        report(40, "sampling shape")
        # Shape gets a wide 35pp (full) / 55pp (mesh) band for per-step progress.
        # Texture sampling (full only) and decode are compressed into the remainder.
        shape_end = 75 if generate_texture else 95
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat(
                cond_512, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params,
                step_callback=lambda c, t: report(40 + round(c / max(t, 1) * (shape_end - 40)), "sampling shape"),
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat(
                cond_1024, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params,
                step_callback=lambda c, t: report(40 + round(c / max(t, 1) * (shape_end - 40)), "sampling shape"),
            )
            res = 1024
        elif pipeline_type == '1024_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1024,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                step_callback=lambda c, t: report(40 + round(c / max(t, 1) * (shape_end - 40)), "sampling shape"),
            )
        elif pipeline_type == '1536_cascade':
            shape_slat, res = self.sample_shape_slat_cascade(
                cond_512, cond_1024,
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, 1536,
                coords, shape_slat_sampler_params,
                max_num_tokens,
                step_callback=lambda c, t: report(40 + round(c / max(t, 1) * (shape_end - 40)), "sampling shape"),
            )
        # Texture always uses the 1024 model with 1024 conditioning (its training
        # distribution), even when the shape stays at 512. Skipped entirely for
        # mesh-only (white model) generation.
        tex_slat = None
        if generate_texture:
            report(75, "sampling texture")
            tex_slat = self.sample_tex_slat(
                cond_1024, self.models['tex_slat_flow_model_1024'],
                shape_slat, tex_slat_sampler_params,
                step_callback=lambda c, t: report(75 + round(c / max(t, 1) * 15), "sampling texture"),
            )
        self._evict_flow_models()
        torch.cuda.empty_cache()
        decode_start = 90 if generate_texture else 95
        report(decode_start, "decoding mesh" + (" and texture" if generate_texture else ""))
        out_mesh = self.decode_latent(
            shape_slat, tex_slat, res,
            progress_callback=lambda p, s: report(decode_start + round(p * (98 - decode_start) / 100), s),
        )
        report(100, "pipeline complete")
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
