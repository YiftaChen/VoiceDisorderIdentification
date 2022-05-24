from turtle import forward
from architecture.backend.yamnet.model import yamnet
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,dimensions=[],out_dim=1,activation=nn.ReLU(),freeze_backend_grad=True) -> None:
        super().__init__()
        layers = []
        input_dim=1024
        for dimension in dimensions:
            layers += [nn.Linear(input_dim,dimension,bias=False),
                        # nn.BatchNorm1d(num_features=dimension),
                        activation]
            input_dim = dimension
        layers+=[nn.Linear(input_dim,out_dim,bias=False)]
      
        self.classifier=nn.Sequential(*layers)
        self.backend = yamnet(pretrained=True,remove_orig_classifier=True,freeze_grad=freeze_backend_grad)
        self.full_layout = nn.Sequential(self.backend,self.classifier)

    def forward(self,x):
        return self.full_layout(x).squeeze()
