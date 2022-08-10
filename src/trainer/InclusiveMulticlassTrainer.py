from trainer.BaseTrainer import BaseTrainer,BatchResult
import torch
import torch.nn as nn
from math import prod

class MulticlassTrainer(BaseTrainer):
    def __init__(self, datasets, model, optimizer, hyper_params, early_stop=float('inf'), device=None, verbose=False, logResults=True) -> None:
        super().__init__(datasets, model, optimizer, hyper_params, early_stop, device, verbose, logResults)     
        self.criterion = nn.CrossEntropyLoss()   

    def train_batch(self, sample) -> BatchResult:        
        x = sample['data'].to(device=self.device)
        y = sample['classification'].float().squeeze().to(device=self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs,y)
        loss.backward()
        self.optimizer.step()
        
        predictions = outputs.detach()>0

        len_predictions = prod(y.shape)
        accuracy = torch.sum(predictions==y)/len_predictions      
        per_class_accuracy = torch.sum(predictions==y,axis=0)/y.shape[0]
        return BatchResult(predictions,loss,accuracy,per_class_accuracy)

    def validate_batch(self, sample) -> BatchResult:
        x = sample['data'].to(device=self.device)
        y = sample['classification'].float().squeeze().to(device=self.device)
        accuracy = 0.0
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.criterion(outputs,y)            
            predictions = outputs.detach()>0
            len_predictions = prod(y.shape)
            accuracy = torch.sum(predictions==y)/len_predictions             
            per_class_accuracy = torch.sum(predictions==y,axis=0)
        return BatchResult(predictions,loss,accuracy,per_class_accuracy)
