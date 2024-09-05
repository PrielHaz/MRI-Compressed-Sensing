# File: models/vanilla.py

import torch

from models.subsampling import SubsamplingLayer


class VanillaModel(torch.nn.Module):
    def __init__(self, drop_rate, device, learn_mask):
        super().__init__()

        self.subsample = SubsamplingLayer(
            drop_rate, device, learn_mask
        )  # initialize subsampling layer - use this in your own model

        # 1 input channel, maybe MRI have 1 channel and not like RGB.
        # Outputs 1 channel - the reconstructed image.
        # padding same - to keep the same size of the image after convolution, so that reconstruction have same height and width
        # as the original image. In 3x3 kernel, padding 1 will be added to each side of the image.
        self.conv = torch.nn.Conv2d(1, 1, 3, padding="same").to(
            device
        )  # some conv layer as a naive reconstruction model - you probably want to find something better.

    def forward(self, x, return_noised=False):
        noised_in_image_domain = self.subsample(
            x
        )  # * get subsampled input in image domain - use this as first line in your own model's forward
        # Cause we can access only the output of this layer, which is in image domain and after removing frequencies.

        x = self.conv(noised_in_image_domain).squeeze(1)
        if return_noised:
            return x, noised_in_image_domain.squeeze(1)
        return x
