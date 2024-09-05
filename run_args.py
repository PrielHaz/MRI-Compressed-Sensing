# File: run_args.py

import getpass
import os

import torch


class RunArgs:
    def __init__(
        self,
        data_path=None,
        results_root=None,
        seed=0,  # piazza said dont touch the seed
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=16,
        num_workers=1,  # dont touch it, remain 1, makes problems.
        num_epochs=50,
        report_interval=100,  # num of batches to train before visulize etc
        drop_rate=0.8,
        learn_mask=False,
        lr=0.01,
        mask_lr=0.01,
        val_test_split=0.3,
        early_stopping=2,
        scale=0.3,  # scale the lr by this factor every epoch
        scale_mask_lr=0.2,  # will sclae the mask lr by this factor every epoch
        start_from_cp=None,  # if path - will load that cp and the epoch idx and run from there.
    ):
        assert device == "cuda"
        if data_path is None:
            if getpass.getuser() == "priel":
                data_path = "/home/priel/Downloads/deepProj/fastmri_knee/"
            elif getpass.getuser() == "priel.hazan":
                data_path = "/mnt/cslash2/priel/fastmri_knee/"
            elif getpass.getuser() in ["mben-zaquen", "bzmao"]:
                data_path = "/datasets/fastmri_knee/"
            else:
                raise ValueError("Please specify data path in run_args.py")
        this_files_dir_abs_path = os.path.dirname(os.path.abspath(__file__))
        # if checkpoints_path is None:
        #     checkpoints_path = os.path.join(this_files_dir_abs_path, "checkpoints")
        # self.checkpoints_path = checkpoints_path
        if results_root is None:
            if getpass.getuser() == "priel":
                results_root = (
                    "/home/priel/Downloads/deepProj/DL-MRI-CompressedSensing/results/"
                )
            elif getpass.getuser() == "priel.hazan":
                results_root = "/mnt/cslash2/priel/deepProject/results/"
            elif getpass.getuser() in ["mben-zaquen", "bzmao"]:
                results_root = "results"
            else:
                raise ValueError("Please specify results root path in run_args.py")
        self.results_root = results_root
        self.seed = seed
        self.data_path = data_path
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.report_interval = report_interval
        self.drop_rate = drop_rate
        self.learn_mask = learn_mask
        self.results_root = results_root
        self.lr = lr
        self.mask_lr = mask_lr
        self.val_test_split = val_test_split
        self.early_stopping = early_stopping
        self.scale = scale
        self.scale_mask_lr = scale_mask_lr
        self.start_from_cp = start_from_cp

    def __str__(self):
        attrs = vars(self)
        return "\n".join(
            f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in attrs.items()
        )


if __name__ == "__main__":
    args = RunArgs()
    print(args)
