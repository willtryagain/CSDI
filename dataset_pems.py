import os
import pickle
import re

import numpy as np
import pandas as pd
import scipy.io as sio
from icecream import ic
from torch.utils.data import DataLoader, Dataset

# dataloader
# dataset


class PemsSF(Dataset):
    def __init__(self, indices=None, seed=1, mode="train"):
        if indices is not None:
            self.indices = np.array(indices)
        else:
            self.indices = None
        np.random.seed(seed)
        self.num_missing = 140
        # B x T x N
        self.observed_values = sio.loadmat("data/pems/nrtsi.mat")["pems_" + mode]
        self.eval_length = self.observed_values.shape[1]
        # pick 140 time steps randomly
        self.observed_masks = ~np.isnan(self.observed_values)
        self.gt_masks = self.observed_masks.copy()
        for i in range(self.observed_values.shape[0]):
            missing_list = np.random.permutation(self.observed_values.shape[1])[
                : self.num_missing
            ]
            self.gt_masks[i, missing_list, :] = False
        self.gt_masks = self.gt_masks.astype(np.float32)
        self.observed_values = np.nan_to_num(self.observed_values)
        self.observed_masks = self.observed_masks.astype(np.float32)

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]
        return {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }

    def __len__(self) -> int:
        if self.indices is not None:
            return self.indices.shape[0]
        else:
            return self.observed_values.shape[0]


def get_dataloader(seed=1, batch_size=16):
    data = sio.loadmat("data/pems/nrtsi.mat")["pems_train"]
    indices = np.arange(data.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices = indices[: int(0.9 * data.shape[0])]
    val_indices = indices[int(0.9 * data.shape[0]) :]

    train_dataset = PemsSF(seed=seed, mode="train", indices=train_indices)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    val_dataset = PemsSF(seed=seed, mode="train", indices=val_indices)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    test_dataset = PemsSF(seed=seed, mode="test")
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_dataloader, val_dataloader, test_dataloader
