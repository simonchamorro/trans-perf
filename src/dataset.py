import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class PerfDataset(Dataset):
    def __init__(self, x, y, length):
        pad = length - x.shape[1]
        tensor_x = torch.from_numpy(x).float()
        self.x = F.pad(tensor_x, (0, pad, 0, 0), 'constant', 0)
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]