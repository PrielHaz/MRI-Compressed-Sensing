import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_and_plot(done_dir_path, match_string, output_dir_path):
    data_with_mask = {
        "drop_rate": [],
        "train_mean_psnr": [],
        "test_mean_psnr": [],
        "train_std_psnr": [],
        "test_std_psnr": [],
    }
    data_without_mask = {
        "drop_rate": [],
        "train_mean_psnr": [],
        "test_mean_psnr": [],
        "train_std_psnr": [],
        "test_std_psnr": [],
    }

    for dir_name in os.listdir(done_dir_path):
        match = re.match(match_string, dir_name)
        if match:
            exp_number = match.group(1)
            drop_rate = float(match.group(2))
            mask_lr = match.group(3)
            mask_lr = None if mask_lr == "NA" else float(mask_lr)

            dir_path = os.path.join(done_dir_path, dir_name)

            # if there is a file named dont_include in the directory, skip it
            if os.path.exists(os.path.join(dir_path, "dont_include")):
                continue

            try:
                with open(os.path.join(dir_path, "train_results.json"), "r") as file:
                    train_results = json.load(file)
                with open(os.path.join(dir_path, "test_results.json"), "r") as file:
                    test_results = json.load(file)
            except FileNotFoundError:
                print(f"JSON files not found in directory: {dir_name}")
                continue

            if mask_lr is None:
                data_without_mask["drop_rate"].append(drop_rate)
                data_without_mask["train_mean_psnr"].append(train_results["mean_psnr"])
                data_without_mask["test_mean_psnr"].append(test_results["mean_psnr"])
                data_without_mask["train_std_psnr"].append(train_results["std_psnr"])
                data_without_mask["test_std_psnr"].append(test_results["std_psnr"])
            else:
                data_with_mask["drop_rate"].append(drop_rate)
                data_with_mask["train_mean_psnr"].append(train_results["mean_psnr"])
                data_with_mask["test_mean_psnr"].append(test_results["mean_psnr"])
                data_with_mask["train_std_psnr"].append(train_results["std_psnr"])
                data_with_mask["test_std_psnr"].append(test_results["std_psnr"])

    def plot_data(data, title_prefix):
        sorted_indices = np.argsort(data["drop_rate"])
        sorted_drop_rate = np.array(data["drop_rate"])[sorted_indices]

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(
            sorted_drop_rate,
            np.array(data["train_mean_psnr"])[sorted_indices],
            label="Train",
            color="blue",
            marker="o",
            linestyle="-",
            markersize=8,
        )
        plt.plot(
            sorted_drop_rate,
            np.array(data["test_mean_psnr"])[sorted_indices],
            label="Test",
            color="green",
            marker="o",
            linestyle="--",
            markersize=8,
        )
        plt.title(f"{title_prefix} Mean PSNR vs Drop Rate")
        plt.xlabel("Drop Rate")
        plt.ylabel("Mean PSNR")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        # For PSNR Std plot
        plt.plot(
            sorted_drop_rate,
            np.array(data["train_std_psnr"])[sorted_indices],
            label="Train",
            color="red",
            marker="o",
            linestyle="-",
            markersize=8,
        )
        plt.plot(
            sorted_drop_rate,
            np.array(data["test_std_psnr"])[sorted_indices],
            label="Test",
            color="purple",
            marker="o",
            linestyle="--",
            markersize=8,
        )
        plt.title(f"{title_prefix} PSNR Std vs Drop Rate")
        plt.xlabel("Drop Rate")
        plt.ylabel("PSNR Std")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        output_file = os.path.join(
            output_dir_path, f"{title_prefix.replace(' ', '_')}_PSNR.png"
        )
        plt.savefig(output_file)
        plt.close()

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    if data_with_mask["drop_rate"]:
        plot_data(data_with_mask, "With Learning Mask")
    if data_without_mask["drop_rate"]:
        plot_data(data_without_mask, "Without Learning Mask")


# Usage
done_dir_name = "DoneDirV2"  # * Change only this
model_name = "UNet_2Plus"
# model_name = "UNET" # old model
done_dir_path = f"/mnt/cslash2/priel/deepProject/results/{done_dir_name}/"
match_string = r"exp(\d+)_drop([\d\.]+)_mask_lr(NA|\d+\.?\d*)_model_" + model_name
output_dir_path = f"./psnrs_graphs/{done_dir_name}/"
parse_and_plot(done_dir_path, match_string, output_dir_path)
