from typing import List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torchvision.models as models


class VGG19Features(nn.Module):
    def __init__(self, device: torch.device):
        super(VGG19Features, self).__init__()
        self.device = device

        self.vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vgg19 = self.vgg19.to(device)
        for param in self.vgg19.parameters():
            param.requires_grad = False
        for name, layer in self.vgg19.features._modules.items():
            logging.info(f"{name}: {layer}")

        self.content_layer = "21"
        self.style_layers = ["0", "5", "10", "19", "28"]

    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x.to(self.device)

        content_features: Optional[torch.Tensor] = None
        style_features: List[torch.Tensor] = []

        for name, layer in self.vgg19.features._modules.items():
            x = layer(x)  # type: ignore FIXME: stronger typing required
            if name == self.content_layer:
                content_features = x.clone().cpu()
            if name in self.style_layers:
                style_features.append(x.clone().cpu())
        assert content_features is not None, "content features not found"

        return content_features, style_features
