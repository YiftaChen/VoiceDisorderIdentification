from numpy import zeros
import torch
import torch.nn as nn
import architecture.backend.sincNET.model as sn

from torchsummary import summary


class Basic1DCNN(nn.Module):
    def __init__(self,channels,kernelsize,strides,mlpLayers,outputDim,dropout=0.3,poolEvery=2) -> None:
      super().__init__()

      input_dim = 50000

      layers = []

      prevChannel = 1

      for idx,(channel,kernelSize,stride) in enumerate(zip(channels,kernelsize,strides)):
        if (idx%poolEvery==0 and idx!=0):
          layers.append(nn.MaxPool1d(16,8))

        layers.append(nn.Conv1d(prevChannel,channel,kernelSize,stride))
        layers.append(nn.BatchNorm1d(channel))        
        layers.append(nn.LeakyReLU())      
        layers.append(nn.Dropout1d(dropout))  
        
        prevChannel = channel

      self.cnn = nn.Sequential(*layers)

      cnnOutputSize = self.getCnnOutputDim(input_dim)

      layers = []

      prevSize = cnnOutputSize
      for mlpLayerSize in mlpLayers:
        layers.append(nn.Linear(prevSize,mlpLayerSize))
        layers.append(nn.LeakyReLU())
        prevSize = mlpLayerSize

      layers.append(nn.Linear(prevSize,outputDim))

      self.mlp = nn.Sequential(*layers)
      a=5


    def getCnnOutputDim(self, input_dim):
      input = torch.zeros(input_dim).unsqueeze(0).unsqueeze(0)
      output = self.cnn(input)
      return output.numel()

    def forward(self,x):
      x = self.cnn(x)
      x = x.reshape(x.size(0),-1)
      return self.mlp(x).squeeze()








class SincNETClassifier(nn.Module):
    def __init__(self,printSummary=True) -> None:
        super().__init__()
        
        input_dim = 50000
        SincNetOptions = {'input_dim': input_dim,
          'fs': 50000, # sample rate
          'cnn_N_filt': [4,8,16], # channels (first channel is sinc_conv others are just 1d_conv)
          'cnn_len_filt': [256,5,5], # filter length
          'cnn_max_pool_len': [3,3,3],
          'cnn_use_laynorm_inp': True,
          'cnn_use_batchnorm_inp': False,
          'cnn_use_laynorm':[True,True,True],
          'cnn_use_batchnorm':[False,False,False],
          'cnn_act': ["relu","relu","relu"],
          'cnn_drop':[0.0,0.0,0.0]
          }

        self.model = sn.SincNet(SincNetOptions)
        
        MLPOptions = {'input_dim': self.model.out_dim,
          'fc_lay': [128,64,1],
          'fc_drop': [0.0,0.0,0.0],           
          'fc_use_laynorm_inp': True,
          'fc_use_batchnorm_inp':False,
          'fc_use_batchnorm': [True,True,False],
          'fc_use_laynorm': [False,False,False],
          'fc_act': ['leaky_relu','leaky_relu','linear']
          }

        self.classifier = sn.MLP(MLPOptions)

        self.full_layout = nn.Sequential(self.model,self.classifier)

        # if (printSummary):
        #     summary(self.full_layout,torch.Size((1,input_dim)),device='cpu')

    def forward(self,x):
        x=x.squeeze(1)
        return self.full_layout(x).squeeze()
        


