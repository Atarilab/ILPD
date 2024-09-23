import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pinocchio as pin

try:
    from data.IL_MPC_dataset import MPCTrajectoryDataset
except:
    from learning.data.IL_MPC_dataset import MPCTrajectoryDataset


def get_dataloaders(
    data_dir,
    batch_size,
    use_set_points:bool,
    contact_conditioned:bool,
    normalize:bool,
    ):
       
    data_path = os.path.join(data_dir, "data.npz")
    assert os.path.exists(data_path), "Data not found"

    dataset = MPCTrajectoryDataset(data_path, use_set_points, contact_conditioned, normalize)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    batch = next(iter(train_dataloader))
    
    print("Number of samples:")
    print("Train:", len(train_dataset))
    print("Test:", len(test_dataset))
    print()

    print("Train batch shape:")
    for key, value in batch.items():
        print(key, ":", list(value.shape))

    return train_dataloader, test_dataloader

if __name__ == "__main__":
    train_dataloader, test_dataloader = get_dataloaders("/home/atari_ws/data/trot", 32, True, True)
    print(len(train_dataloader.dataset))