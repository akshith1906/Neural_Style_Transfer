import logging
from typing import List, Tuple
from enum import Enum
import time

import torch
import torch.optim as optim
import wandb

from src.model import VGG19Features
from src.loss import get_content_loss, get_style_loss


class Optimizer(Enum):
    LBFGS = "lbfgs"
    ADAM = "adam"


def _get_losses(
    model: VGG19Features,
    generated_image: torch.Tensor,
    target_content_feat: torch.Tensor,
    target_style_feats: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    content, style_feats = model(generated_image)

    content_loss = get_content_loss(content, target_content_feat)

    style_loss = 0
    for target_style_feat, style_feat in zip(target_style_feats, style_feats):
        style_loss += get_style_loss(style_feat, target_style_feat)
    assert type(style_loss) is torch.Tensor

    return content_loss, style_loss


def _log_losses(
    step: int,
    num_steps: int,
    model: VGG19Features,
    generated_image: torch.Tensor,
    target_content: torch.Tensor,
    target_style_features: List[torch.Tensor],
) -> None:
    content_loss, style_loss = _get_losses(
        model, generated_image, target_content, target_style_features
    )
    wandb.log({"content_loss": content_loss.item(), "style_loss": style_loss.item()})

    if (step % (num_steps // min(10, num_steps)) != 0) and (step != num_steps - 1):
        return
    print(
        f"step {step + 1} -> content loss: {content_loss.item()}, style loss: {style_loss.item()}"
    )


def _transfer_lbfgs(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    model: VGG19Features,
    content_weight: float,
    style_weight: float,
    num_steps: int,
):
    target_content_feat, _ = model(content_image)
    _, target_style_feats = model(style_image)

    generated_image = content_image.clone().requires_grad_(True)
    optimizer = optim.LBFGS([generated_image])

    def lbfgs_closure():
        optimizer.zero_grad()

        with torch.no_grad():
            generated_image.clamp_(0, 1)

        content_loss, style_loss = _get_losses(
            model, generated_image, target_content_feat, target_style_feats
        )
        loss = content_weight * content_loss + style_weight * style_loss
        loss.backward()

        return loss

    for i in range(num_steps):
        start_time = time.time()
        optimizer.step(lbfgs_closure)
        _log_losses(
            i,
            num_steps,
            model,
            generated_image,
            target_content_feat,
            target_style_feats,
        )
        time_taken = time.time() - start_time
        print(f"\ttime taken: {time_taken:.2f} seconds")

    return generated_image.detach().squeeze(0)


def _transfer_adam(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    model: VGG19Features,
    content_weight: float,
    style_weight: float,
    num_steps: int,
):
    target_content_feat, _ = model(content_image)
    _, target_style_feats = model(style_image)

    generated_image = content_image.clone().requires_grad_(True)
    optimizer = optim.Adam([generated_image], lr=0.1)

    for i in range(num_steps):
        start_time = time.time()

        optimizer.zero_grad()

        with torch.no_grad():
            generated_image.clamp_(0, 1)

        content_loss, style_loss = _get_losses(
            model, generated_image, target_content_feat, target_style_feats
        )
        loss = content_weight * content_loss + style_weight * style_loss
        loss.backward()

        optimizer.step()

        _log_losses(
            i,
            num_steps,
            model,
            generated_image,
            target_content_feat,
            target_style_feats,
        )
        time_taken = time.time() - start_time
        logging.info(f"\ttime taken: {time_taken:.2f} seconds")

    return generated_image.detach()


def transfer_style(
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    model: VGG19Features,
    content_weight: float,
    style_weight: float,
    num_steps: int,
    optimizer: Optimizer,
) -> torch.Tensor:
    print("TRAIN INFO:")
    print(f"\tcontent_weight: {content_weight}")
    print(f"\tstyle_weight: {style_weight}")
    print(f"\tnum_steps: {num_steps}")
    print(f"\toptimizer: {optimizer}")
    content_image = content_image.unsqueeze(0)
    style_image = style_image.unsqueeze(0)

    kwargs = {
        "content_image": content_image,
        "style_image": style_image,
        "model": model,
        "content_weight": content_weight,
        "style_weight": style_weight,
        "num_steps": num_steps,
    }
    match optimizer:
        case Optimizer.LBFGS:
            return _transfer_lbfgs(**kwargs).squeeze(0)
        case Optimizer.ADAM:
            return _transfer_adam(**kwargs).squeeze(0)
