from turtle import forward
from unicodedata import bidirectional

from numpy import inner
from architecture.backend.yamnet.model import yamnet,Identity
import torch
import torch.nn as nn
import s3prl.hub 
from transformers import Wav2Vec2Processor, HubertModel
import torchaudio.pipelines
import math
import torch.nn.functional as F

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

class FullyConnectedClassificationHead(nn.Module):
    def __init__(self,input_dim,out_dim,activation=nn.ReLU()):
        super().__init__()
        layers=[nn.Linear(input_dim,1280,bias=False),nn.BatchNorm1d(num_features=1280),activation,nn.Linear(1280,out_dim,bias=False)]
        self.layers = nn.Sequential(*layers)
    def forward(self,x):
        x = x.reshape(x.shape[0],-1)
        return self.layers(x)

class ConvolutionalClassificationHead(nn.Module):
    def __init__(self,input_dim,out_dim,activation=nn.ReLU(), kernels=[(3,5)]*3,strides=(1,2),inner_dim = 16,initial_image_dim=(49,768)):
        super().__init__()
        layers = [] 
        inp = input_dim
        h_dim,w_dim = initial_image_dim
        inner_dim = [inner_dim]*len(kernels)
        strides = [strides]*len(kernels)

        for kernel,in_dim,stride in zip(kernels,inner_dim,strides):
            layers += [nn.Conv2d(inp,in_dim,kernel_size=kernel,stride=stride),nn.BatchNorm2d(in_dim),activation]
            inp=in_dim
            stride_x,stride_y = stride
            kernel_x,kernel_y = kernel
            h_dim = math.floor((h_dim-kernel_x)/stride_x)+1
            w_dim = math.floor((w_dim-kernel_y)/stride_y)+1
        layers_fully_connected = [nn.Linear(h_dim*w_dim*inp,512,bias=False),nn.BatchNorm1d(num_features=512),activation,nn.Linear(512,out_dim,bias=False)]
        self.layers = nn.Sequential(*layers)
        self.layers_fully_connected = nn.Sequential(*layers_fully_connected)
    def forward(self,x):
        x = x.unsqueeze(-1).reshape(x.shape[0],1,x.shape[1],x.shape[2])
        # assert False, f"shape of x is {x.shape}"
        x = self.layers(x)
        out = x.reshape(x.shape[0],-1)
        return self.layers_fully_connected(out)

class LSTMClassificationHead(nn.Module):
    def __init__(self,out_dim,bidirectional=True,layer_count=2,dropout=0.5,hidden_dim=100) -> None:
        super().__init__()
        self.LSTM=nn.LSTM(input_size=768,hidden_size=hidden_dim,num_layers = layer_count,batch_first=True,bidirectional=bidirectional,dropout=dropout)
        self.bidirectional = bidirectional
        if bidirectional:
            self.Linear = nn.Linear(100*4,out_dim,bias=False)
        else:
            self.Linear = nn.Linear(100,out_dim,bias=False)
            
    def forward(self,x):
        x,(hn,cn) = self.LSTM(x)
        if self.bidirectional:
            x = torch.cat((x[:,0,:],x[:,-1,:]),dim=1)
        else:
            x = x[:,-1,:]
        x = self.Linear(x)
        return x
class VGGClassificationHead(nn.Module):
    def __init__(self,out_dim,initial_image_dim=(49,768)):
        super().__init__()
        self.net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11_bn')
        self.net.features[0] = nn.Conv2d(1,64,(3,3),padding=(1,1))
        self.net.classifier[6] = nn.Linear(4096,out_dim,False)
        # assert False, f"self.net is {self.net}" 
    def forward(self,x):
        # assert False,f"shape of x is {x.shape}"
        x = x.unsqueeze(-1).reshape(x.shape[0],1,x.shape[1],x.shape[2])
        x = F.pad(x,(0,0,88,88),"constant",0)
        return self.net(x)


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
    def __init__(self,hp,configuration="base",out_dim=1,activation=SinusoidalActivation(),freeze_backend_grad=True) -> None:
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
        self.input_dim = input_dim*49
        self.classifier = LSTMClassificationHead(1,bidirectional=hp["bidirectional"],
                                                layer_count=hp["lstm_layer_count"],
                                                hidden_dim=hp["hidden_dim"],
                                                dropout=hp["dropout"])
        # self.classifier.classifier[6] = nn.Linear(4096,out_dim,False)
        # assert False, f"classifier {self.classifier}"
        # self.classifier=ConvolutionalClassificationHead(1,out_dim,activation=activation,
        #                                                 kernels=hp['classification_head_kernels'],
        #                                                 strides = hp['classification_head_strides'],
        #                                                 inner_dim=hp['classification_inner_dim'])
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
        # assert False, f"x shape {x.shape}"
        # x = self.classifier(x)
        # x = x.reshape(x.shape[0],-1)
        x =  self.classifier(x)
        return x.squeeze()

class HubertMulticlassClassifier(HubertClassifier):
    def __init__(self,hp,configuration="base",out_dim=1,activation=SinusoidalActivation(),freeze_backend_grad=True,num_classes = 11) -> None:
        super().__init__(hp,configuration,out_dim,activation,freeze_backend_grad)
        classifiers = []
        for class_id in range(num_classes):
          classifiers+=[FullyConnectedClassificationHead(self.input_dim,out_dim=1)]
        self.classifier = nn.ModuleList(classifiers)


    def forward(self,x):
        x = x.reshape((x.shape[0],x.shape[-1]))
        x = torchaudio.functional.resample(x, 50000, self.bundle.sample_rate)        
        x,y = self.model(x)
        # assert False, f"x shape {x.shape}"
        # x = self.classifier(x)
        # x = x.reshape(x.shape[0],-1)
        classifications = [classifier(x) for classifier in self.classifier]
        
        classifications = torch.cat(classifications,axis=1)
        return classifications