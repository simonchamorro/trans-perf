import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F


class PerfDataset(Dataset):
    def __init__(self, x, y, length):
        pad = length - x.shape[1]
        tensor_x = F.pad(torch.from_numpy(x).float(), (0, pad, 0, 0), 'constant', 0)
        self.x = torch.zeros((tensor_x.shape[0], tensor_x.shape[1], 2))
        pos_enc = torch.Tensor(np.arange(0,1,1/self.x.shape[1]))
        pos_enc = pos_enc.unsqueeze(0).repeat((self.x.shape[0],1))
        self.x[:,:,0] = tensor_x
        self.x[:,:,1] = pos_enc
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]