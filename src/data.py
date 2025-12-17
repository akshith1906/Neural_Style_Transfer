import logging
import os
from typing import Tuple
import random

from PIL import Image
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision.transforms as transforms


def _preprocess(imsize: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(imsize),
            transforms.ToTensor(),
        ]
    )


def _load_image(image_path: str, size: int = 512) -> torch.Tensor:
    image = Image.open(image_path)
    processed_image = _preprocess(size)(image)

    assert type(processed_image) is torch.Tensor
    return processed_image


def get_base_images(
    content_dir_path: str, style_dir_path: str
) -> Tuple[torch.Tensor, str, torch.Tensor, str]:
    conte_files = [f for f in os.listdir(content_dir_path)]
    conte_image_name = random.choice(conte_files)
    logging.info(f"loading image {conte_image_name} as content source")
    conte_image = _load_image(os.path.join(content_dir_path, conte_image_name))

    style_files = [f for f in os.listdir(style_dir_path)]
    style_image_name = random.choice(style_files)
    logging.info(f"loading image {style_image_name} as style source")
    style_image = _load_image(os.path.join(style_dir_path, style_image_name))

    return conte_image, conte_image_name, style_image, style_image_name


def visualise_images(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    generated_image: torch.Tensor,
    run_name: str,
) -> None:
    def to_np(image: torch.Tensor) -> numpy.ndarray:
        return image.clamp(0, 1).permute(1, 2, 0).numpy()

    content_image_np = to_np(content_image)
    style_image_np = to_np(style_image)
    generated_image_np = to_np(generated_image)

    _, ax = plt.subplots(1, 3)
    ax[0].imshow(content_image_np)
    ax[0].axis("off")
    ax[0].set_title("Content Image")

    ax[1].imshow(style_image_np)
    ax[1].axis("off")
    ax[1].set_title("Style Image")

    ax[2].imshow(generated_image_np)
    ax[2].axis("off")
    ax[2].set_title("Generated Image")

    # save the image
    plt.savefig(f"{run_name}.png")
