# File: trainer.py
import gc
import json
import os
import random
import sys

import numpy as np
import PIL
import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

import models.model as model_file
from models.vanilla import VanillaModel
from run_args import RunArgs
from utils.myUtils import (
    CompositeLoss,
    MS_SSIMLoss,
    MSELoss,
    PSNRLoss,
    calc_psnrs,
    clear_memory,
    get_first_group_lr,
    get_main_py_dir_abs_path,
    get_next_experiment_number,
    plot_list,
    print_groups_lr,
    psnr,
    save_psnrs_results,
    scale_learning_rate,
    set_learning_rate,
    visualize_reconstructions,
)
from utils.utils import create_data_loaders, freq_to_image


class Trainer:

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        args,
        stop_epoch_after_num_batches,
        exp_num,
        experiment_dir,
        figures_dir,
        is_run_in_notebook,
        dir_to_save_images_abs_path,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.num_image_to_plot = 4

        self.best_val_mean_loss = float("inf")
        self.best_checkpoint_pt_path = None
        self.no_improvement_epochs = 0

        self.stop_epoch_after_num_batches = stop_epoch_after_num_batches

        self.train_loader, self.validation_loader, self.test_loader = (
            create_data_loaders(args)
        )

        self.exp_num = exp_num
        self.experiment_dir = experiment_dir
        self.figures_dir = figures_dir

        self.is_run_in_notebook = is_run_in_notebook
        self.dir_to_save_images_abs_path = dir_to_save_images_abs_path

    @staticmethod
    def update_desc(phase, epoch, loss, batch_idx, lr, args):
        return f"{phase} Epoch: {epoch}, Loss: {loss:.4f}, Batch: {batch_idx}, Lr: {lr}, mask_lr: {args.mask_lr}"

    def train(self, epoch_idx):
        clear_memory(self.args.device)
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Progress",
            file=sys.stdout,
        )

        # calculate lr_in_epoch and mask_lr_in_epoch
        lr_in_epoch = self.args.lr
        mask_lr_in_epoch = self.args.mask_lr
        for _ in range(epoch_idx):
            lr_in_epoch *= self.args.scale
            if self.args.learn_mask:
                mask_lr_in_epoch *= self.args.scale_mask_lr
        set_learning_rate(self.optimizer, lr_in_epoch)
        print(
            f"Training with lr_in_epoch: {lr_in_epoch}, mask_lr_in_epoch: {mask_lr_in_epoch}"
        )

        train_losses = []
        self.model.train()
        # freq_images shape: torch.Size([16, 320, 320, 2]), target_images shape: torch.Size([16, 320, 320])
        for batch_idx, (freq_images, target_images) in progress_bar:
            freq_images = freq_images.to(self.args.device)
            target_images = target_images.to(self.args.device)
            self.optimizer.zero_grad()
            reconstructed_images = self.model(freq_images)
            # print(
            #     "reconstructed_images.shape=", reconstructed_images.shape
            # )  # [16, 320, 320]
            # print("target_images.shape=", target_images.shape)
            loss = self.criterion(reconstructed_images, target_images)
            loss.backward()
            self.optimizer.step()
            train_losses.append(loss.item())
            if self.args.learn_mask:
                self.model.subsample.mask_grad(mask_lr_in_epoch)

            if batch_idx % self.args.report_interval == 0:
                visualize_reconstructions(
                    self.model,
                    self.test_loader,
                    self.args,
                    self.dir_to_save_images_abs_path,
                    epoch_idx,
                    batch_idx,
                    num_images=self.num_image_to_plot,
                    is_run_in_notebook=self.is_run_in_notebook,
                )
            progress_bar.set_description(
                self.update_desc(
                    "Train",
                    epoch_idx,
                    loss.item(),
                    batch_idx,
                    get_first_group_lr(self.optimizer),
                    self.args,
                )
            )
            progress_bar.update(1)
            if (
                self.stop_epoch_after_num_batches is not None
                and batch_idx >= self.stop_epoch_after_num_batches - 1
            ):
                break
        del freq_images, target_images, reconstructed_images, loss
        progress_bar.close()
        plot_list(
            train_losses,
            f"Training epoch {epoch_idx} mean losses vs batch idx. lr in epoch: {get_first_group_lr(self.optimizer)}",
            "Batch idx",
            "Mean Losses",
            os.path.join(self.figures_dir, f"train_losses_epoch_{epoch_idx}.png"),
        )

    def validate(self, epoch_idx):
        print("^^^^^^^^^^^^^^^^^ Training done. Validation phase: ^^^^^^^^^^^^^^^^^")
        clear_memory(self.args.device)
        progress_bar = tqdm(
            enumerate(self.validation_loader),
            total=len(self.validation_loader),
            desc="Progress",
            file=sys.stdout,
        )
        self.model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_images = 0
            for batch_idx, (freq_images, target_images) in progress_bar:
                freq_images, target_images = freq_images.to(
                    self.args.device
                ), target_images.to(self.args.device)
                total_val_images += target_images.shape[0]
                outputs = self.model(freq_images)
                loss = self.criterion(outputs, target_images)
                total_val_loss += loss.item()
                progress_bar.set_description(
                    self.update_desc(
                        "Val",
                        epoch_idx,
                        loss.item(),
                        batch_idx,
                        get_first_group_lr(self.optimizer),
                        self.args,
                    )
                )
                progress_bar.update(1)
            progress_bar.close()
            mean_val_loss = total_val_loss / total_val_images
            print(
                f"f{total_val_loss=}, {total_val_images=}, therefore {mean_val_loss=}"
            )

        if mean_val_loss < self.best_val_mean_loss:
            print(
                f"mean_val_loss improved from {self.best_val_mean_loss} to {mean_val_loss}!!! Save checkpoint..."
            )
            self.best_val_mean_loss = mean_val_loss
            self.no_improvement_epochs = 0
            # Save checkpoint
            checkpoint_dir = os.path.join(
                self.experiment_dir,
                f"epoch_{epoch_idx}",
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.best_checkpoint_pt_path = os.path.join(
                checkpoint_dir, f"cp_epoch_{epoch_idx}.pt"
            )
            torch.save(
                {
                    "epoch": epoch_idx,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_mean_loss": mean_val_loss,
                    "curr_lr": get_first_group_lr(self.optimizer),
                    "args": self.args,
                },
                self.best_checkpoint_pt_path,
            )
        else:
            self.no_improvement_epochs += 1
            print(
                f"$$$$$$ Not improved in {epoch_idx=} cause {mean_val_loss=}, {self.best_val_mean_loss=}, so {self.no_improvement_epochs=}"
            )
            # early stopping handled in fit() method.

    def test(self):
        # * when calc psnr - they said skimage, see piazza what they decided on the data range and all: https://piazza.com/class/lqnh9hti76g6od/post/127
        # # So need iterate train\test, in each batch calc min\max of batch, and send max-min as the data_range to myUtils.psnr.
        print("^^^^^^^^^^^^^^^^^ Training done. Test phase: ^^^^^^^^^^^^^^^^^")
        clear_memory(self.args.device)
        print("Loading best checkpoint from: ", self.best_checkpoint_pt_path)
        best_cp = torch.load(self.best_checkpoint_pt_path)
        # Load the best model
        self.model.load_state_dict(best_cp["model_state_dict"])
        self.model.to(self.args.device)
        self.model.eval()
        train_psnrs = calc_psnrs(self.train_loader, self.model, self.args.device)
        clear_memory(self.args.device)
        test_psnrs = calc_psnrs(self.test_loader, self.model, self.args.device)
        print("Saving results to experiment directory...")
        save_psnrs_results(
            train_psnrs,
            self.experiment_dir,
            "train_results",
        )
        save_psnrs_results(
            test_psnrs,
            self.experiment_dir,
            "test_results",
        )
        print(f"Train PSNR Mean: {np.mean(train_psnrs)}, Std: {np.std(train_psnrs)}")
        print(f"Test PSNR Mean: {np.mean(test_psnrs)}, Std: {np.std(test_psnrs)}")

    def fit(self):
        starting_epoch_idx = 0
        if self.args.start_from_cp is not None:
            print("Loading checkpoint from: ", self.args.start_from_cp)
            checkpoint = torch.load(self.args.start_from_cp)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            starting_epoch_idx = checkpoint["epoch"] + 1
            self.best_val_mean_loss = checkpoint["val_mean_loss"]
            self.no_improvement_epochs = 0  # it's ok, let's start from 0
            self.best_checkpoint_pt_path = self.args.start_from_cp
            print(
                f"Loaded checkpoint. Starting epoch: {starting_epoch_idx}, best_val_mean_loss: {self.best_val_mean_loss}"
            )
            # save file named "loaded_checkpoint.txt" in the experiment directory
            with open(
                os.path.join(self.experiment_dir, "loaded_checkpoint.txt"), "w"
            ) as f:
                f.write(f"Loaded checkpoint from: {self.args.start_from_cp}")
            del checkpoint
            gc.collect()

        for epoch_idx in range(starting_epoch_idx, self.args.num_epochs):
            print(f"In fit method, starting epoch_idx: {epoch_idx}")
            self.train(epoch_idx)
            self.validate(epoch_idx)
            if self.no_improvement_epochs >= self.args.early_stopping:
                print(f"Early stopping in after ending {epoch_idx=} ")
                break
            else:
                print(
                    f"Continue to next epoch after ending {epoch_idx=} cause {self.no_improvement_epochs=} < {self.args.early_stopping=}"
                )
        self.test()
