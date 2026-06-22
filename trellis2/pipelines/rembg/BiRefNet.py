from typing import *
import os

from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForImageSegmentation
import torch
import warnings
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self._is_rmbg_14 = "BriaRMBG" in (config.architectures or [])
        self._warmed = False

        # BiRefNet's remote model code still imports compatibility shims from
        # timm.models.*.  timm 1.x supports them but emits a FutureWarning for
        # each import.  Keep the filter local to remote-code loading so genuine
        # deprecation warnings elsewhere remain visible.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Importing from timm\.models\.layers is deprecated.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Importing from timm\.models\.registry is deprecated.*",
                category=FutureWarning,
            )
            if self._is_rmbg_14:
                # RMBG-1.4 targets an older Transformers release whose custom
                # PreTrainedModel cannot complete the current from_pretrained
                # finalization. Construct it from config and load the local
                # safetensors checkpoint directly, as the ComfyUI node does.
                self.model = AutoModelForImageSegmentation.from_config(
                    config, trust_remote_code=True
                )
                checkpoint = os.path.join(model_name, "model.safetensors")
                self.model.load_state_dict(load_file(checkpoint), strict=True)
            else:
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    model_name, trust_remote_code=True
                )
        self.model.eval()
        normalize = (
            transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
            if self._is_rmbg_14
            else transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            )
        )
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    @property
    def warmed(self) -> bool:
        return self._warmed

    @property
    def warmup_source(self) -> str:
        example_path = os.path.join(self.model_name, "example_input.jpg")
        return example_path if os.path.isfile(example_path) else "synthetic RGB image"

    def warmup(self) -> bool:
        """Run one representative inference. Return False if already warm."""
        if self._warmed:
            return False
        example_path = os.path.join(self.model_name, "example_input.jpg")
        if os.path.isfile(example_path):
            with Image.open(example_path) as image:
                warmup_image = image.convert("RGB")
        else:
            warmup_image = Image.new("RGB", (1024, 1024), (127, 127, 127))
        self(warmup_image)
        return True
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            # The service enables cudnn.benchmark for TRELLIS's fixed-shape
            # convolutions. RMBG-1.4 has many distinct convolution shapes, for
            # which first-run algorithm search takes seconds longer than the
            # inference itself. Disable benchmarking only for this forward.
            with torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ):
                outputs = self.model(input_images)
            if self._is_rmbg_14:
                pred = outputs[0][0][0, 0].cpu()
            else:
                pred = outputs[-1].sigmoid()[0].squeeze().cpu()
        self._warmed = True
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
