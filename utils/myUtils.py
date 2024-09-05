# File: utils/myUtils.py

import gc
import json
import math
import os
import sys

import numpy as np
import piq
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt

# from pytorch_msssim import ssim
from skimage.metrics import peak_signal_noise_ratio as psnr_skimage
from tqdm import tqdm


# Function to manually update the learning rate
def scale_learning_rate(optimizer, scale):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * scale


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def print_groups_lr(optimizer):
    for idx, param_group in enumerate(optimizer.param_groups):
        group_name = param_group.get("name", f"Group {idx+1}")
        print(f"Group name: {group_name}, lr: {param_group['lr']}")
    print()


def get_first_group_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def plot_list(lst, title, xlabel, ylabel, save_path):
    plt.plot(lst)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def clear_memory(device):
    if torch.cuda.is_available():
        # Get the total memory
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9

        print(f"Total memory on {device}: {total_memory:.2f} GB")
        # Get initial memory stats
        initial_stats = torch.cuda.memory_stats(device)
        initial_used = initial_stats["allocated_bytes.all.peak"] / 1e9
        initial_free = total_memory - initial_used

        # Print memory usage before cleanup
        print(
            f"before cleanup: Memory on {device}: {initial_used:.2f} GB used, {initial_free:.2f} GB free, "
        )

        # Clearing cache and garbage collection
        torch.cuda.empty_cache()  # Clear PyTorch's CUDA cache
        gc.collect()  # Clear Python's garbage collection

        # Get memory stats after cleanup
        after_stats = torch.cuda.memory_stats(device)
        after_used = after_stats["allocated_bytes.all.peak"] / 1e9
        after_free = total_memory - after_used
        freed = initial_used - after_used

        # Print memory usage after cleanup
        print(
            f"after cleanup(freed {freed:.2f} GB): Memory on {device}: {after_used:.2f} GB used, {after_free:.2f} GB free, "
        )


def get_main_py_dir_abs_path():
    # its the abs path of the father of father dir of this file.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def psnr(ground_truth, noisy_image, data_range=None):
    return psnr_skimage(ground_truth, noisy_image, data_range=data_range)


def calc_psnrs(data_loader, model, device, user_description=""):
    def calc_description(user_description, batch_idx):
        return f"{user_description} - batch {batch_idx}"

    psnrs = []
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=calc_description(user_description, 0),
            file=sys.stdout,
        )

        for batch_idx, (freq_images, target_images) in progress_bar:
            freq_images, target_images = freq_images.to(device), target_images.to(
                device
            )
            outputs = model(freq_images)
            batch_data_range = torch.max(target_images) - torch.min(target_images)
            for target_image, output_image in zip(target_images, outputs):
                image_psnr = psnr(
                    target_image.cpu().numpy(),
                    output_image.cpu().numpy(),
                    data_range=batch_data_range.item(),
                )
                psnrs.append(image_psnr)
            progress_bar.set_description(calc_description(user_description, batch_idx))
            progress_bar.update(1)
    return psnrs


def save_psnrs_results(psnrs, dir_path, base_name):
    """ "
    save the psnrs list as a numpy array as join(dir_path, base_name + '.npy')
    and the std and mean as a json file as join(dir_path, base_name + '.json')
    """
    mean_psnr = np.mean(psnrs)
    std_psnr = np.std(psnrs)
    np.save(os.path.join(dir_path, base_name + ".npy"), psnrs)
    results = {"mean_psnr": mean_psnr, "std_psnr": std_psnr}
    with open(os.path.join(dir_path, base_name + ".json"), "w") as f:
        json.dump(results, f, indent=4)


# My implementation.
# def psnr(img1, img2, max_val=255.0):
#     """Peak Signal to Noise Ratio
#     img1 and img2 have range [0, 255]"""  # so I think need rescale to [0, 255] before use it
#     # if got numpy array, convert to tensor
#     if isinstance(img1, np.ndarray):
#         img1 = torch.from_numpy(img1)
#     if isinstance(img2, np.ndarray):
#         img2 = torch.from_numpy(img2)
#     mse = torch.mean((img1 - img2) ** 2, dtype=torch.float32)
#     return 20 * torch.log10(max_val / torch.sqrt(mse))


class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):  # can get sum too
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, img1, img2):
        return self.mse(img1, img2)


class PSNRLoss(nn.Module):
    def __init__(self, max_val=1.0, epsilon=1e-8):
        super(PSNRLoss, self).__init__()
        self.max_val = max_val
        self.epsilon = epsilon

    def forward(self, inp, target):
        """
        inp: input image, size (B, H, W)
        target: target image, size (B, H, W)
        """
        mses_of_batch = torch.mean((inp - target) ** 2, dim=(1, 2))
        # eps for numerical stability
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mses_of_batch + self.epsilon))
        # Negative PSNR to make it a loss (lower is better).
        # the divide by 20 is to normalize the loss to be approx in range [0, 4].
        # * need to print for mri images, see scale, to weight it good with other losses.
        psnr_losses = 4 - psnr / 20
        return psnr_losses.mean()


# multi-scale structural similarity index. tested
class MS_SSIMLoss(nn.Module):
    def __init__(
        self,
        device,
        add_mse,
        ms_ssim_factor=10.0,
        data_range: float = 1.0,
        reduction: str = "mean",
        target_min=-2.1,
        target_max=16.6842,
        eps=1e-3,
    ):
        super(MS_SSIMLoss, self).__init__()
        self.data_range = data_range
        self.reduction = reduction
        self.target_min = torch.tensor(target_min).to(device)
        self.target_max = torch.tensor(target_max).to(device)
        self.eps = eps
        # Can use also piq.SSIMLoss
        self.msssim = piq.MultiScaleSSIMLoss(
            data_range=data_range, reduction=reduction
        ).to(device)
        self.mse = nn.MSELoss(reduction=reduction)
        self.add_mse = add_mse
        self.ms_ssim_factor = ms_ssim_factor

    def forward(self, inp, target):
        """
        inp: input image, size (B, C, H, W) or (B, H, W)
        target: target image, size (B, C, H, W) or (B, H, W)
        """
        torch.autograd.set_detect_anomaly(True)
        # Directly return the MS-SSIM loss computed by the piq library, they aready do the 1-ms_ssim.
        # maybe mul by some factor(see prints to determine) so that same scale as other losses.

        # if got 3 dims, add channel dim
        if len(inp.shape) == 3:
            inp = inp.unsqueeze(1)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)

        # add 4 to both images, to assure min pixel value is 0. min pixel value of target is -2.0002 but the reconstruction can maybe be less..
        # print(f"inp min berfore: {inp.min()}, target min before: {target.min()}")

        val_to_add_to_all_pixels = 10
        inp += val_to_add_to_all_pixels
        target += val_to_add_to_all_pixels

        final_loss = self.ms_ssim_factor * self.msssim(inp, target)
        to_print = f"ms_ssim: {final_loss:.5f}"
        if self.add_mse:
            mse_loss = self.mse(inp, target)
            to_print += f", mse: {mse_loss:.5f}"
            final_loss += mse_loss
        print(to_print)
        return final_loss


class CompositeLoss(nn.Module):
    def __init__(
        self, device, mse_weight=1.0, ms_ssim_weight=1.0, mae_weight=1.0, data_range=8.0
    ):
        super(CompositeLoss, self).__init__()
        self.device = device
        # total = mse_weight + ms_ssim_weight + mae_weight

        # self.mse_weight = mse_weight / total
        # self.ms_ssim_weight = ms_ssim_weight / total
        # self.mae_weight = mae_weight / total

        self.mse_loss = MSELoss()
        self.mae_loss = torch.nn.L1Loss()
        self.ms_ssim_loss = MS_SSIMLoss(device, data_range=data_range)

    def forward(self, inp, target_images):
        mse = self.mse_loss(inp, target_images)
        mae = self.mae_loss(inp, target_images)
        ms_ssim = self.ms_ssim_loss(inp, target_images)
        # print(
        #     f"mse: {mse} with weight: {self.mse_weight} || mae: {mae} with weight: {self.mae_weight} || ms_ssim: {ms_ssim} with weight: {self.ms_ssim_weight}"
        # )
        # total_loss = (
        #     self.mse_weight * mse
        #     + self.mae_weight * mae
        #     + self.ms_ssim_weight * ms_ssim
        # )
        total_loss = mse + mae + ms_ssim
        return total_loss


def visualize_reconstructions(
    model,
    dataloader,
    args,
    dir_to_save_images,
    epoch_idx,
    batch_idx,
    num_images=4,
    is_run_in_notebook=False,
):
    """
    Visualize and save images during model evaluation.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): The device to run the model on.
        dir_to_save_images (str): Directory to save images.
        epoch_idx (int): Current epoch index.
        drop_rate (float): Dropout rate used in the model.
        num_images (int): Number of images to visualize and save.
    """
    with torch.no_grad():
        model.eval()
        freq_images, target_images = next(iter(dataloader))
        freq_images = freq_images.to(args.device)
        target_images = target_images.to(args.device)
        reconstructed_images, noised_images = model(freq_images, return_noised=True)
        reconstructed_images = (
            reconstructed_images.squeeze()
        )  # remove all dimensions of size 1

        # Print shapes of images for debugging
        print(
            f"\n\n --------------------- Saving images epoch {epoch_idx}, trained until: {batch_idx} ---------------------\n\n"
        )
        # Visualize images
        fig, axs = plt.subplots(num_images, 3, figsize=(10, 10))
        for i in range(num_images):
            axs[i, 0].imshow(target_images[i].cpu(), cmap="gray")
            axs[i, 0].set_title(f"Target {i}")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(noised_images[i].cpu(), cmap="gray")
            axs[i, 1].set_title(f"Noised {i}, drop rate: {args.drop_rate}")
            axs[i, 1].axis("off")
            axs[i, 2].imshow(reconstructed_images[i].cpu(), cmap="gray")
            axs[i, 2].set_title(f"Reconstructed {i}")
            axs[i, 2].axis("off")
        plt.tight_layout()
        if is_run_in_notebook:
            plt.show()
        else:
            # not run in notebook.
            if dir_to_save_images is not None:
                plt.savefig(
                    f"{dir_to_save_images}/epoch_{epoch_idx}_batchIdx_{batch_idx}_drop_{args.drop_rate}.png"
                )
        plt.close()


def get_next_experiment_number(file_path):
    try:
        with open(file_path, "r") as file:
            exp_num = int(file.read().strip()) + 1
    except (FileNotFoundError, ValueError):
        exp_num = 1  # Set to 1 if file does not exist or is empty

    with open(file_path, "w") as file:
        file.write(str(exp_num))
    return exp_num


if __name__ == "__main__":
    ms_loss = MS_SSIMLoss(data_range=1.0, reduction="mean")
    # generate random batch of 1x320x320
    inp = torch.rand((16, 1, 320, 320), requires_grad=False)
    inp -= 3
    target = torch.rand((16, 1, 320, 320), requires_grad=False)
    loss = ms_loss(inp, target)
    print(loss)
