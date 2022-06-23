from turtle import forward
from architecture.backend.yamnet.model import yamnet
import torch
import torch.nn as nn
import s3prl.hub 
from transformers import Wav2Vec2Processor, HubertModel
import torchaudio.pipelines

class YamnetClassifier(nn.Module):
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

class DistilHUBERTClassifier(nn.Module):
    def __init__(self,dimensions=[],out_dim=1,activation=nn.GELU(),freeze_backend_grad=True) -> None:
        super().__init__()
        layers = []
        input_dim = 768
        input_dim = 156*input_dim

        # for dimension in dimensions:
        #     layers += [nn.Linear(input_dim,dimension,bias=False),
        #                 # nn.BatchNorm1d(num_features=dimension),
        #                 activation]
        #     input_dim = dimension
        layers+=[nn.Linear(input_dim,1280,bias=False),nn.BatchNorm1d(num_features=1280),activation,nn.Linear(1280,out_dim,bias=False)]
      
        self.classifier=nn.Sequential(*layers)
        self.backend = s3prl.hub.distilhubert()    
        if freeze_backend_grad:
            for param in self.backend.parameters():
                param.requires_grad = False
            for param in self.backend.model.model.output_layer.parameters():
                param.requires_grad = True
            for param in self.backend.model.model.encoder.layers[1].parameters():
                param.requires_grad = True
            for param in self.backend.model.model.encoder.layers[0].parameters():
                param.requires_grad = True

        self.full_layout = nn.Sequential(self.backend,self.classifier)

    def forward(self,x):
        if(len(x)>2):
            x=x.squeeze()
        x = self.backend(x)['last_hidden_state']
        x = x.reshape(x.shape[0],-1)
        return self.classifier(x).squeeze()

class SinusoidalActivation(nn.Module):
    def forward(self,x):
        return torch.sin(x)

class Wav2Vec2Classifier(nn.Module):
    def __init__(self,dimensions=[],configuration="base",out_dim=1,activation=nn.ReLU(),freeze_backend_grad=True) -> None:
        super().__init__()
        layers = []
        if configuration == "base":
            input_dim = 768
            self.bundle = torchaudio.pipelines.WAV2VEC2_BASE
        elif configuration == "large":
            input_dim = 1024
            self.bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        self.model = self.bundle.get_model()

        # for dimension in dimensions:
        #     layers += [nn.Linear(input_dim,dimension,bias=False),
        #                 # nn.BatchNorm1d(num_features=dimension),
        #                 activation]
        #     input_dim = dimension
        input_dim = input_dim*49

        layers+=[nn.Linear(input_dim,1280,bias=False),nn.BatchNorm1d(num_features=1280),activation,nn.Linear(1280,out_dim,bias=False)]
      
        self.classifier=nn.Sequential(*layers)
        # self.backend = s3prl.hub.hubert()    
        if freeze_backend_grad:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            # for param in self.processor.parameters():
            #     param.requires_grad = False


    def forward(self,x):
        x = x.reshape((x.shape[0],x.shape[-1]))
        x = torchaudio.functional.resample(x, 50000, self.bundle.sample_rate)        
        x,y = self.model(x)
        x = x.reshape(x.shape[0],-1)
        return self.classifier(x).squeeze()

class HubertClassifier(nn.Module):
    def __init__(self,dimensions=[],configuration="base",out_dim=1,activation=nn.ReLU(),freeze_backend_grad=True) -> None:
        super().__init__()
        layers = []
        if configuration == "base":
            input_dim = 768
            self.bundle = torchaudio.pipelines.HUBERT_BASE
        elif configuration == "large":
            input_dim = 1024
            self.bundle = torchaudio.pipelines.HUBERT_LARGE
        elif configuration == "xlarge":
            input_dim = 1280
            self.bundle = torchaudio.pipelines.HUBERT_XLARGE

        self.model = self.bundle.get_model()

        # for dimension in dimensions:
        #     layers += [nn.Linear(input_dim,dimension,bias=False),
        #                 # nn.BatchNorm1d(num_features=dimension),
        #                 activation]
        #     input_dim = dimension
        input_dim = input_dim*49

        layers+=[nn.Linear(input_dim,1280,bias=False),nn.BatchNorm1d(num_features=1280),activation,nn.Linear(1280,out_dim,bias=False)]
      
        self.classifier=nn.Sequential(*layers)
        # self.backend = s3prl.hub.hubert()    
        if freeze_backend_grad:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            # for param in self.processor.parameters():
            #     param.requires_grad = False


    def forward(self,x):
        x = x.reshape((x.shape[0],x.shape[-1]))
        x = torchaudio.functional.resample(x, 50000, self.bundle.sample_rate)        
        x,y = self.model(x)
        x = x.reshape(x.shape[0],-1)
        return self.classifier(x).squeeze()