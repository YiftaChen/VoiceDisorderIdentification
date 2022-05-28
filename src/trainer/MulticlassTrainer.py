from trainer.BaseTrainer import BaseTrainer,BatchResult
import torch
import torch.nn as nn

class MulticlassTrainer(BaseTrainer):
    def __init__(self, dataset, model, optimizer, hyper_params, early_stop=float('inf'), device=None, verbose=False, logResults=True) -> None:
        super().__init__(dataset, model, optimizer, hyper_params, early_stop, device, verbose, logResults)     
        self.criterion = nn.CrossEntropyLoss()   

    def train_batch(self, sample) -> BatchResult:        
        x = sample['data'].to(device=self.device)
        y = sample['classification'].long().squeeze().to(device=self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x)
        # print(outputs)
        # print(y)
        loss = self.criterion(outputs,y)
        loss.backward()
        self.optimizer.step()
        
        predictions = outputs.argmax(1)

        len_predictions = 1 if len(predictions.shape) == 0 else len(predictions)
        accuracy = torch.sum(predictions==y)/len_predictions      

        return BatchResult(loss,accuracy)

    def validate_batch(self, sample) -> BatchResult:
        x = sample['data'].to(device=self.device)
        y = sample['classification'].long().squeeze().to(device=self.device)
        accuracy = 0.0
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.criterion(outputs,y)            
            predictions = outputs.argmax(1)
            len_predictions = 1 if len(predictions.shape) == 0 else len(predictions)
            accuracy = torch.sum(predictions==y)/len_predictions             
         
        return BatchResult(loss,accuracy)
