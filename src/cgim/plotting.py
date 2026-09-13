from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class _ClassificationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        _, _, logits = self.model(inputs)
        return logits


class _BinaryLogitTarget:
    def __init__(self, target_class):
        self.target_class = target_class

    def __call__(self, model_output):
        logit = model_output[0]
        return logit if self.target_class == 1 else -logit


def save_gradcam(
    model,
    image_path,
    device,
    output_prefix,
    formats=("svg", "png"),
):
    """Create the binary-class Grad-CAM overlay used for interpretation."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise ImportError(
            "Grad-CAM requires the optional plotting dependencies: "
            "python -m pip install -e '.[plots]'"
        ) from exc

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda tensor: tensor
                if tensor.shape[0] == 3
                else tensor.expand(3, -1, -1)
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    original_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        _, _, logits = model(input_tensor)
    logit = logits.squeeze().item()
    probability = torch.sigmoid(torch.tensor(logit)).item()
    predicted_class = int(logit >= 0)

    wrapped_model = _ClassificationWrapper(model).to(device).eval()
    target_layer = model.encoder[7][1].conv2
    cam = GradCAM(model=wrapped_model, target_layers=[target_layer])
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[_BinaryLogitTarget(predicted_class)],
    )[0]

    rgb_image = np.asarray(original_image.resize((224, 224))).astype(np.float32) / 255.0
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(5, 5))
    plt.imshow(visualization)
    plt.axis("off")
    plt.tight_layout()
    saved_paths = []
    for file_format in formats:
        normalized = file_format.lower().lstrip(".")
        if normalized not in {"svg", "png"}:
            raise ValueError("Plot formats must be 'svg' and/or 'png'")
        path = output_prefix.with_suffix(f".{normalized}")
        figure.savefig(
            path,
            format=normalized,
            dpi=300 if normalized == "png" else None,
            bbox_inches="tight",
            pad_inches=0,
        )
        saved_paths.append(path)
    plt.close(figure)
    return {
        "logit": logit,
        "probability": probability,
        "predicted_class": predicted_class,
        "paths": saved_paths,
    }
