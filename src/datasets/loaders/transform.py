import torch
from torch.nn import functional as F

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        wav,sample_rate, classification = sample['data'],sample['sampling_rate'],sample['classification']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if not isinstance(classification,bool):
            classification = torch.Tensor([classification]).to(torch.int64)
            classification = F.one_hot(classification,10)
        return {'wav': torch.from_numpy(wav),
                'sampling_rate': sample_rate,
                'classification':classification}
