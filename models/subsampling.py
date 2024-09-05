# File: models/subsampling.py

import torch

from data import transforms

RES = 320  # used resolution (image size to which we crop data) - for our purposes it's constant


class SubsamplingLayer(torch.nn.Module):
    def __init__(self, drop_rate, device, learn_mask):
        super().__init__()

        self.learn_mask = learn_mask

        self.drop_rate = 1 - drop_rate  # Percentage of the input to keep.

        # Initialize a random mask and its binary version for the dropping process.
        self.mask = torch.randn(1, RES, RES, 2).to(device)
        self.binary_mask = self.mask.clone().to(device)

        self.mask = torch.nn.Parameter(self.mask, requires_grad=False)
        self.binary_mask = torch.nn.Parameter(
            self.binary_mask, requires_grad=learn_mask
        )

    def apply_mask(self, x):
        # Calculate the threshold value to keep `drop_rate` percentage of data.
        # takes the most important k pixels, according to mask values(between 1 and -1).
        # in backward, if learned - we update the non learnable mask real values by the gradient of the binary mask.
        drop_threshold = torch.topk(
            self.mask.flatten(),
            int(self.drop_rate * len(self.mask.flatten())),
            largest=True,
        ).values.min()

        # Update the binary mask without tracking gradients, to drop part of the data.
        with torch.no_grad():
            self.binary_mask.data = (self.mask >= drop_threshold).to(torch.float)

        return self.binary_mask * x

    def mask_grad(self, lr):

        if self.learn_mask:
            # print("updating mask with lr=", lr)
            mask_before_update = self.mask.clone()
            # print(f"mask before update: {self.mask}")
            self.mask -= lr * self.binary_mask.grad
            self.mask.clamp(-1, 1)
            # print(f"mask after update: {self.mask}")
            # print(
            #     f"sum of squared diffs between masks: {torch.sum((self.mask - mask_before_update) ** 2)}"
            # )

        return

    def forward(self, x):
        x = self.apply_mask(
            x
        )  # apply mask over input, removing part of frequency domain data

        # the inverse fourier transform (ifft) will return the subsampled image to the image domain,
        # then the last dim which is size 2 will be removed by change every complex number(cause ifft matrix have them) to real number
        return transforms.complex_abs(transforms.ifft2_regular(x)).unsqueeze(
            1
        )  # cast to image domain
