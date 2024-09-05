# File: data/mri_dataset.py

import pathlib
import random
import h5py
from torch.utils.data import Dataset
from data import transforms
import numpy as np

"""Dataset class and data transformations"""


class SliceData(Dataset):
    def __init__(self, root, transform, split, validation=False):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            split (float, optional): A float between 0 and 1. This controls what fraction
                of the test set goes to validation.
        """
        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if split < 1:
            # Note val-test split is not randomized here for comparison across submissions. In real-world it should be.
            num_files = int(len(files) * split)
            files = files[:num_files] if validation else files[num_files:]
        for fname in sorted(files):
            if ".cache" in str(fname):
                continue
            with h5py.File(fname, "r") as data:
                if data.attrs["acquisition"] == "CORPD_FBK":
                    kspace = data["kspace"]
                    num_slices = kspace.shape[0]
                    self.examples += [
                        (fname, slice) for slice in range(5, num_slices - 2)
                    ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, "r") as data:
            kspace = data["kspace"][slice]
            target = (
                data["reconstruction_rss"][slice]
                if "reconstruction_rss" in data
                else None
            )
            return self.transform(kspace, target, data.attrs, fname.name, slice)


class DataTransform:
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        kspace is the image in frequency domain.
        We move to image domain to normalize, and then go back to freq domain because that's where
        we'll be working on the image (subsampling it).

        target is the original image we'd like to reconstruct (ground truth)

        """
        kspace = transforms.to_tensor(kspace)
        image = transforms.ifft2_regular(kspace)
        image = transforms.complex_center_crop(
            image, (self.resolution, self.resolution)
        )

        image, mean, std = transforms.normalize_instance(image, eps=1e-11)

        normalized_kspace = transforms.fft2(image)

        target = transforms.to_tensor(target)
        target, mean, std = transforms.normalize_instance(target, eps=1e-11)

        return normalized_kspace, target
