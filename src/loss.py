import torch
import torch.nn.functional as F


def get_content_loss(
    content_features: torch.Tensor, generated_features: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(content_features, generated_features)


def _gram_matrix(features: torch.Tensor) -> torch.Tensor:
    B, C, H, W = features.size()
    assert B == 1, "here, we assume a batch size of 1"
    features = features.view(C, H * W)

    gram = torch.mm(features, features.t())
    gram /= C * H * W
    return gram


def get_style_loss(
    style_features: torch.Tensor, generated_features: torch.Tensor
) -> torch.Tensor:
    gram_style = _gram_matrix(style_features)
    gram_gener = _gram_matrix(generated_features)

    return F.mse_loss(gram_gener, gram_style)
