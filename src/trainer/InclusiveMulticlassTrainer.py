from trainer.BaseTrainer import BaseTrainer,BatchResult
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sn
import core
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_curve, PrecisionRecallDisplay, average_precision_score
# from math import prod

class MulticlassTrainer(BaseTrainer):
    def __init__(self, datasets, model, optimizer, hyper_params, early_stop=float('inf'), device=None, verbose=False, logResults=True, criterion=None,classes_amount=11, confusion_mat_every=5) -> None:
        super().__init__(datasets, model, optimizer, hyper_params, early_stop, device, verbose, logResults,confusion_mat_every)   
        self.classes_amount = classes_amount
        if criterion == None:
            self.criterion = nn.CrossEntropyLoss()   
        else:
            self.criterion = criterion    

    def train_batch(self, sample) -> BatchResult:        
        x = sample['data'].to(device=self.device)
        y = sample['classification'].float().squeeze().to(device=self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs,y)
        loss.backward()
        self.optimizer.step()
        
        predictions = outputs>0

        len_predictions = torch.numel(y)
        accuracy = torch.sum(predictions==y)/len_predictions      
        return BatchResult(predictions.cpu().detach().numpy(),outputs.cpu().detach().numpy(),loss,accuracy)

    def validate_batch(self, sample) -> BatchResult:
        x = sample['data'].to(device=self.device)
        y = sample['classification'].float().squeeze().to(device=self.device)
        accuracy = 0.0
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.criterion(outputs,y)            
            predictions = outputs>0
            len_predictions = torch.numel(y)
            accuracy = torch.sum(predictions==y)/len_predictions             
            return BatchResult(predictions.cpu().detach().numpy(),outputs.cpu().detach().numpy(),loss,accuracy)

    def log_confusion_matrix(self, valid_pred, valid_gt, epoch):        
        classes = list(core.params.PathologiesToIndex.keys())
        cf_matrix = multilabel_confusion_matrix(valid_gt, valid_pred)
        # assert False, f"shape of cf_matrix {cf_matrix.shape}"
        for c in range(cf_matrix.shape[0]):
            df_cm = pd.DataFrame(cf_matrix[c], index = ["Not Sick GT","Sick GT"],
                                columns = ["Not Sick Pred","Sick Pred"])
            plt.figure(figsize = (12,7))
            sn.heatmap(df_cm, annot=True,fmt='g')
            os.makedirs(core.params.project_dir + f'/src/confusion_matrices',exist_ok=True)
            plt.savefig(core.params.project_dir + f'/src/confusion_matrices/output_{epoch}_{classes[c]}.png')
            plt.close('all') 

    def log_precision_recall(self, valid_scores, valid_gt, epoch):      
        valid_proba_t = nn.functional.sigmoid(torch.Tensor(valid_scores))
        valid_gt_t = np.array(valid_gt)
        classes = list(core.params.PathologiesToIndex.keys())
        for i in range(self.classes_amount):
            prec, rec, thresholds = precision_recall_curve(valid_gt_t[:,i], valid_proba_t[:,i])
            avg_prec = average_precision_score(valid_gt_t[:,i], valid_proba_t[:,i])         

            os.makedirs(core.params.project_dir + f'/src/precision_recalls',exist_ok=True)
            display = PrecisionRecallDisplay(recall=rec, precision=prec, average_precision=avg_prec)
            display.plot().figure_.savefig(core.params.project_dir + f'/src/precision_recalls/output_{epoch}_{classes[i]}.png')   


    def process_valid_results(self, valid_pred, valid_scores, valid_gt, epoch):
        self.log_precision_recall(valid_scores, valid_gt, epoch)

        if epoch % self.confusion_mat_every == 0:
            self.log_confusion_matrix(valid_pred, valid_gt, epoch)
