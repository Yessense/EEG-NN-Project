import torch
from torch.utils.data import TensorDataset, DataLoader


def create_dataset(data, target):
    raw, fft = data
    assert len(raw) == len(fft)

    dataset = TensorDataset(torch.tensor([raw]).float().squeeze(),
                            torch.tensor([fft]).float().squeeze(),
                            torch.tensor([target]).long().squeeze())
    return dataset


def create_loaders(data, bs=64, jobs=0):
    trn_ds, val_ds = data
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    return trn_dl, val_dl
