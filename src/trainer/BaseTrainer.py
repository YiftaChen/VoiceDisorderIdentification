from abc import ABC,abstractclassmethod
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import math
from typing import NamedTuple
import numpy as np
from ray import tune
import os
import core.params
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
class BatchResult(NamedTuple):   
    predictions: torch.BoolTensor 
    loss: float
    accuracy: float
    per_class_accuracy: torch.FloatTensor

class BaseTrainer(object):
    def __init__(self,datasets,model,optimizer,hyper_params,early_stop=float('inf'),device=None,verbose=False,logResults=True) -> None:
        # self.dl = dataloader
        self.train_set, self.val_set, self.test_set = datasets        
        self.train_set =  DataLoader(
            self.train_set,
            batch_size=hyper_params['train_batch_size'],
            shuffle=True,
            num_workers=hyper_params['num_workers']
        )
        self.val_set =  DataLoader(
            self.val_set,
            batch_size=hyper_params['vald_batch_size'],
            shuffle=True,
            num_workers=hyper_params['num_workers']
        )
        self.test_set =  DataLoader(
            self.test_set,
            batch_size=hyper_params['test_batch_size'],
            shuffle=True,
            num_workers=hyper_params['num_workers']
        )
        self.early_stop = early_stop
        self.disableTQDM = not verbose
        self.model = model
        self.optimizer = optimizer
        self.logResults = logResults
        
        self.hp = hyper_params
        if (device is not None):
            self.device = device
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"        
        self.model = self.model.to(device=self.device)

    def train_val_test_split(self,ds):
        ds_len =  len(ds)
        len_test = math.floor(ds_len*0.1)
        len_valid = math.floor(ds_len*0.1)
        len_train = ds_len-len_test-len_valid              
        return torch.utils.data.random_split(ds, [len_train,len_valid,len_test]) 

    def train_batch(self, sample) -> BatchResult:
        pass

    def validate_batch(self, sample) -> BatchResult:
        pass


    def train(self):
        train_losses = []        
        train_accuracies = []
      
        last_acc=0
        runs_without_improv=0

        with tqdm(range(self.hp['epochs'])) as pbar_epochs:
            for idx,epoch in enumerate(pbar_epochs):
                running_loss = 0.0
                epoch_accuracy = 0.0
                epoch_accuracies = []               
                epoch_losses = []
                epoch_class_accuracies = []
                sample_count = 0
                with tqdm(self.train_set,disable=self.disableTQDM) as pbar:
                    for idx,sample in enumerate(pbar): 
                        batchRes = self.train_batch(sample)
                        sample_count += sample['data'].shape[0]
                        accuracy = batchRes.accuracy
                        loss = batchRes.loss
                        class_accuracies = batchRes.per_class_accuracy
                        running_loss+=loss
                        train_accuracies += [accuracy]                       
                        epoch_losses += [loss]
                        epoch_accuracies += [accuracy]                        
                        epoch_class_accuracies += [class_accuracies]
                        pbar.set_description(f"train epoch {epoch}, train loss is {loss.item()}, Accuracy {accuracy*100}% ")
                        if idx == len(self.train_set)-1 :
                            epoch_accuracy = sum(epoch_accuracies)/len(epoch_accuracies)                           
                            epoch_loss = sum(epoch_losses)/len(epoch_losses)
                            epoch_class_accuracies = sum(epoch_class_accuracies)/sample_count
                            pbar.set_description(f"train epoch {epoch}, train loss:{running_loss} , Mean Accuracy:{epoch_accuracy*100}%")
                    

                train_losses += [running_loss]
                vald_accuracies = []   
                vald_losses = []    
                vald_class_accuracies = []    
                vald_predictions = []
                vald_true = []
                vald_sample_count = 0
                vald_loss = 0.0
                with tqdm(self.val_set,disable=self.disableTQDM) as t:
                        for idx,sample in enumerate(t):                            
                            batchRes = self.validate_batch(sample)
                            vald_sample_count += sample['data'].shape[0]
                            accuracy = batchRes.accuracy.cpu()
                            loss = batchRes.loss.cpu()
                            class_accuracies = batchRes.per_class_accuracy
                            vald_predictions.extend(batchRes.predictions.cpu().numpy())
                            vald_true.extend(sample['classification'].cpu().numpy())
                            vald_loss+=loss

                            vald_accuracies += [accuracy]                            
                            vald_losses += [loss]
                            vald_class_accuracies += [class_accuracies]
                            t.set_description(f"validation epoch {epoch}, validation loss is {loss.item()}, Accuracy {accuracy*100}%")            
                if epoch%5==0:
                    classes = list(core.params.PathologiesToIndex.keys())
                    cf_matrix = multilabel_confusion_matrix(vald_true, vald_predictions)
                    # assert False, f"shape of cf_matrix {cf_matrix.shape}"
                    for c in range(cf_matrix.shape[0]):
                        df_cm = pd.DataFrame(cf_matrix[c], index = ["Not Sick Pred","Sick Pred"],
                                            columns = ["Not Sick GT","Sick GT"])
                        plt.figure(figsize = (12,7))
                        sn.heatmap(df_cm, annot=True,fmt='g')
                        os.makedirs(core.params.project_dir + f'/src/confusion_matrices',exist_ok=True)
                        plt.savefig(core.params.project_dir + f'/src/confusion_matrices/output_{epoch}_{classes[c]}.png')


                accuracy = np.array(vald_accuracies).mean()               
                loss = np.array(vald_losses).mean()
                vald_class_acc = sum(vald_class_accuracies) / vald_sample_count
                if (self.logResults):
                    tune.report(valid_acc=accuracy,train_acc=epoch_accuracy.item())                                            
                    tune.report(valid_loss=loss,train_loss=epoch_loss.item())                        

                if (accuracy.item()>last_acc):
                    last_acc=accuracy.item()
                    runs_without_improv=0
                else:
                    runs_without_improv+=1
                # pbar_epochs.set_description(f"validation epoch {epoch} validation class accuracies {vald_class_acc}")            

                pbar_epochs.set_description(f"validation epoch {epoch}, train acc {'{:.2f}'.format(epoch_accuracy.item())}, validation acc {'{:.2f}'.format(accuracy.item())}")            

                if (runs_without_improv>=self.early_stop):
                    vald_losses += [vald_loss]
                    return self.model,train_losses,vald_loss, train_accuracies    

                vald_losses += [vald_loss]

        return self.model,train_losses,vald_losses, train_accuracies,vald_accuracies

    def test(self,model):
        test_loss = 0.0
        test_accuracies = []
        test_precisions = []
        test_recalls = []

        with tqdm(self.test_set) as t:
                for idx,sample in enumerate(t):
                    x = sample['data'].to(device=self.device)
                    y = sample['classification'].float().squeeze().to(device=self.device)
                    with torch.no_grad():
                        outputs = model(x)
                        loss = self.critereon(outputs,y)
                        test_loss += loss.item() 
                        predictions = outputs.detach() > 0
                        len_predictions = 1 if len(predictions.shape) == 0 else len(predictions)
                        accuracy = torch.sum(predictions==y)/len_predictions
                        precision = torch.sum(predictions*(predictions==y))/torch.sum(predictions)
                        recall = torch.sum(predictions*(predictions==y))/torch.sum(y)
                        test_accuracies += [accuracy]
                        test_precisions += [precision]
                        test_recalls += [recall]

                    self.writer.add_scalar('Loss/test',loss.item(),idx)
                    self.writer.add_scalar('Accuracy/test',accuracy,idx)
                    self.writer.add_scalar('Precision/test',precision,idx)
                    self.writer.add_scalar('Recall/test',recall,idx)
                    t.set_description(f"test:, test loss is {loss.item()}, Accuracy {accuracy*100}%, Precision {precision*100}%, Recall {recall*100}%")
        
        test_accuracies = sum(test_accuracies)/len(test_accuracies)
        test_precision = sum(test_precisions)/len(test_precisions)
        test_recall = sum(test_recalls)/len(test_recalls)



        return test_loss,test_accuracies,test_precision,test_recall
