import torch
from torch.nn import functional as F
import torch.nn as nn

class ToOneHot(nn.Module):
    def __call__(self, classification):
        if not isinstance(classification,bool):
            classification = torch.Tensor([classification]).to(torch.int64)
            classification = F.one_hot(classification,10).squeeze()
        return classification

class ToTensor(nn.Module):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample).float()
class SwitchDim(nn.Module):
    def __call__(self, sample):
        return sample[:,:100]