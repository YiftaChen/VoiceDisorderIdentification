from turtle import forward
from architecture.backend.yamnet.model import yamnet
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,dimensions=[],out_dim=1) -> None:
        super().__init__()
        self.backend = yamnet(pretrained=True,remove_classification_layer=True)
        layers = []
        input_dim=1024
        for dimension in dimensions:
            layers += [nn.Linear(input_dim,dimension,bias=False),
                        nn.ReLU()]
            input_dim = dimension
        layers+=[nn.Linear(input_dim,out_dim,bias=False)]
        self.classification = nn.Sequential(*layers)

    def forward(self,x):
        return self.classification(self.backend(x))
