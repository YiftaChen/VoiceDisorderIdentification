from turtle import forward
from architecture.backend.yamnet.model import yamnet
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,dimensions=[],out_dim=1) -> None:
        super().__init__()
        layers = []
        input_dim=1024
        for dimension in dimensions:
            layers += [nn.Linear(input_dim,dimension,bias=False),
                        nn.BatchNorm1d(num_features=dimension),
                        nn.ReLU()]
            input_dim = dimension
        layers+=[nn.Linear(input_dim,out_dim,bias=False)]
      
        self.backend = yamnet(pretrained=True,remove_classification_layer=True)
        self.classification = nn.Sequential(self.backend,*layers)

    def forward(self,x):
        return self.classification(x)
