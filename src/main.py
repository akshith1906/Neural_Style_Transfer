import os
import logging
import argparse

import torch
import wandb

from src.model import VGG19Features
from src.train import transfer_style, Optimizer
from src.data import get_base_images, visualise_images


def _get_proj_name(content_image_path: str, style_image_path: str) -> str:
    return content_image_path + style_image_path


def main(
    content_weight: float, style_weight: float, num_steps: int, optimizer: Optimizer
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"using device: {device}")

    content_path = os.path.join("data", "content")
    style_path = os.path.join("data", "styles")
    content_image, content_image_path, style_image, style_image_path = get_base_images(
        content_path, style_path
    )
    run_name = _get_proj_name(content_image_path, style_image_path)

    model = VGG19Features(device)

    run = wandb.init(
        project="style transfer",
        name=run_name,
        config={
            "content_weight": content_weight,
            "style_weight": style_weight,
            "num_steps": num_steps,
            "optimizer": optimizer.value,
        },
    )

    generated_image = transfer_style(
        content_image,
        style_image,
        model,
        content_weight,
        style_weight,
        num_steps,
        optimizer,
    )

    save_path = os.path.join("res", run_name)
    visualise_images(content_image, style_image, generated_image, save_path)
    wandb.save(save_path)

    run.finish()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s : %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(description="style transfer")
    parser.add_argument(
        "--content_weight", type=float, default=1e-2, help="content_weight"
    )
    parser.add_argument("--style_weight", type=float, default=1e1, help="style_weight")
    parser.add_argument("--num_steps", type=int, default=10, help="num_steps")
    parser.add_argument(
        "--optimizer",
        default=Optimizer.LBFGS,
        dest="optimizer",
        choices=[t.value for t in Optimizer],
    )
    args = parser.parse_args()

    content_weight = args.content_weight
    style_weight = args.style_weight
    num_steps = args.num_steps
    optimizer = Optimizer(args.optimizer)

    main(content_weight, style_weight, num_steps, optimizer)
