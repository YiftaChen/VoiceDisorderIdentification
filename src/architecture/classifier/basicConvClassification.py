import torch
import torch.nn as nn
import architecture.backend.sincNET.model as sn

from torchsummary import summary


class SincNETClassifier(nn.Module):
    def __init__(self,printSummary=True) -> None:
        super().__init__()
        
        input_dim = 50000
        SincNetOptions = {'input_dim': input_dim,
          'fs': 50000, # sample rate
          'cnn_N_filt': [80,60,60], # channels (first channel is sinc_conv others are just 1d_conv)
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
          'fc_lay': [1024,1024,1],
          'fc_drop': [0.0,0.0,0.0],           
          'fc_use_laynorm_inp': True,
          'fc_use_batchnorm_inp':False,
          'fc_use_batchnorm': [True,True,False],
          'fc_use_laynorm': [False,False,False],
          'fc_act': ['leaky_relu','leaky_relu','linear']
          }

        self.classifier = sn.MLP(MLPOptions)

        self.full_layout = nn.Sequential(self.model,self.classifier)

        if (printSummary):
            summary(self.full_layout,torch.Size((1,input_dim)),device='cpu')

    def forward(self,x):
        x=x.squeeze(1)
        return self.full_layout(x).squeeze()
        

model = SincNETClassifier()

