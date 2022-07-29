from pickletools import optimize
import torch
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ray import tune
import numpy as np
from ray.tune.integration.wandb import wandb_mixin
from sklearn.metrics import f1_score
import wandb 
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
class Trainer(object):
    def __init__(self,datasets,model,optimizers,critereon,hyper_params,early_stop=float('inf'),device=None,verbose=False) -> None:
        # self.dl = dataloader
        # torch.multiprocessing.set_start_method('spawn')
        self.train_set, self.val_set, self.test_set = datasets
        self.seed = self.train_set.seed
        torch.manual_seed(self.seed)

        self.writer = SummaryWriter("logs/")
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
        self.optimizers = optimizers
        self.critereon=critereon
        self.hp = hyper_params
        if (device is not None):
            self.device = device
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"        
        self.model = self.model.to(device=self.device)
        from pathlib import Path
        Path(f"{self.hp['checkpoints']}/{self.hp['name']}").mkdir(parents=True, exist_ok=True)

    @wandb_mixin
    def train(self):
        train_losses = []
        vald_losses = []

        train_accuracies = []
        train_precisions = []
        train_recalls = []
        train_f1_scores = []
        last_acc=0
        runs_without_improv=0
        max_validation_acc = 0        
        max_validation_f1 = 0

        with tqdm(range(self.hp['epochs'])) as pbar_epochs:
            for idx,epoch in enumerate(pbar_epochs):
                running_loss = 0.0
                epoch_accuracy = 0.0
                epoch_accuracies = []
                epoch_precisions = []
                epoch_recalls = []
                epoch_losses = []
                epoch_f1_scores = []
                train_preds = []
                train_gt = []
                with tqdm(self.train_set,disable=self.disableTQDM) as pbar:
                    for idx,sample in enumerate(pbar):                
                        # print(f"{self.device}")
                        x = sample['data'].to(device=self.device)
                        y = sample['classification'].float().squeeze().to(device=self.device)
                        self.optimizers.zero_grad()
                        outputs = self.model(x)       
                        train_preds = np.concatenate((train_preds,outputs.cpu().detach().numpy()))
                        train_gt = np.concatenate((train_gt,y.cpu().detach().numpy()))
                        loss = self.critereon(outputs,y)
                        loss.backward()
                        self.optimizers.step()
                        running_loss += loss.item() 
                        predictions = outputs.detach() > 0                    
                        len_predictions = 1 if len(predictions.shape) == 0 else len(predictions)
                        accuracy = torch.sum(predictions==y)/len_predictions      
                        precision = torch.sum(predictions*(predictions==y))/torch.sum(predictions)
                        recall = torch.sum(predictions*(predictions==y))/torch.sum(y)
                        f1 = f1_score(y.cpu(),predictions.cpu())

                        train_accuracies += [accuracy]
                        train_precisions += [precision]
                        train_recalls += [recall]
                        train_f1_scores += [f1]

                        epoch_accuracies += [accuracy]
                        epoch_precisions += [precision]
                        epoch_recalls += [recall]
                        epoch_losses += [loss]
                        epoch_f1_scores += [f1]
                        pbar.set_description(f"train epoch {epoch}, train loss is {loss.item()}, Accuracy {accuracy*100}%, Precision {precision*100}%, Recall {recall*100}%")
                        if idx == len(self.train_set)-1 :
                            epoch_accuracy = sum(epoch_accuracies)/len(epoch_accuracies)
                            epoch_precision = sum(epoch_precisions)/len(epoch_precisions)
                            epoch_recall = sum(epoch_recalls)/len(epoch_recalls)
                            epoch_loss = sum(epoch_losses)/len(epoch_losses)
                            epoch_f1_score = sum(epoch_f1_scores)/len(epoch_f1_scores)
                            pbar.set_description(f"train epoch {epoch}, train loss:{running_loss} , Mean F1 Score {epoch_f1_score}, Mean Accuracy:{epoch_accuracy*100}%, Mean Precision:{epoch_precision*100}%, Mean Recall:{epoch_recall*100}%")

                train_losses += [running_loss]
                vald_accuracies = []
                vald_precisions = []
                vald_recalls = []
                vald_losses = []
                vald_f1_scores = []
                vald_preds = []
                vald_gt = []

                vald_loss = 0.0
                with tqdm(self.val_set,disable=self.disableTQDM) as t:
                        for idx,sample in enumerate(t):
                            x = sample['data'].to(device=self.device)
                            y = sample['classification'].float().squeeze().to(device=self.device)
                            with torch.no_grad():
                                outputs = self.model(x)
                                vald_preds = np.concatenate((vald_preds,outputs.cpu().detach().numpy()))
                                vald_gt = np.concatenate((vald_gt,y.cpu().detach().numpy()))
                                loss = self.critereon(outputs,y)
                                vald_loss += loss.item() 
                                predictions = outputs.detach() > 0
                                len_predictions = 1 if len(predictions.shape) == 0 else len(predictions)
                                accuracy = torch.sum(predictions==y)/len_predictions             
                                precision = torch.sum(predictions*(predictions==y))/torch.sum(predictions)
                                recall = torch.sum(predictions*(predictions==y))/torch.sum(y)
                                f1 = f1_score(y.cpu(),predictions.cpu())

                                vald_accuracies += [accuracy.cpu().item()]
                                vald_precisions += [precision.cpu().item()]
                                vald_recalls += [recall.cpu().item()]
                                vald_losses += [loss.cpu().item()]
                                vald_f1_scores += [f1]
                            t.set_description(f"validation epoch {epoch}, validation loss is {loss.item()}, Accuracy {accuracy*100}% F1 score {f1}, Precision {precision*100}%, Recall {recall*100}%")            

                accuracy = np.array(vald_accuracies).mean()
                precision = np.array(vald_precisions).mean()
                recall = np.array(vald_recalls).mean()
                loss = np.array(vald_losses).mean()
                f1 = np.array(vald_f1_scores).mean()
            
                tune.report(valid_acc=accuracy,train_acc=epoch_accuracy.item())                        
                tune.report(valid_precision=precision,train_precision=epoch_precision.item())                        
                tune.report(valid_recall=recall,train_recall=epoch_recall.item())                        
                tune.report(valid_loss=loss,train_loss=epoch_loss.item())                        
                tune.report(valid_f1=f1,train_f1=epoch_f1_score)                        
                


                if (max_validation_acc < accuracy):
                        wandb.log({"best_epoch_train_confusion_matrix" : wandb.plot.confusion_matrix(
                            y_true=train_gt, preds=train_preds > 0,
                            class_names=["Sick","Healthy"])})
                        wandb.log({"best_epoch_validation_confusion_matrix" : wandb.plot.confusion_matrix(
                            y_true=vald_gt, preds=vald_preds > 0,
                            class_names=["Sick","Healthy"])})
                        torch.save({
                            'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizers.state_dict(),
                                'loss': loss,
                                'accuracy':accuracy,
                                'f1':f1,
                                'seed':self.seed
                                }, f"{self.hp['checkpoints']}/{self.hp['name']}/accuracy_based_model.pt")
                        
                if (max_validation_f1 < f1):
                    wandb.log({"best_epoch_train_confusion_matrix" : wandb.plot.confusion_matrix(
                        y_true=train_gt, preds=train_preds > 0,
                        class_names=["Healthy","Sick"])})
                    wandb.log({"best_epoch_validation_confusion_matrix" : wandb.plot.confusion_matrix(
                        y_true=vald_gt, preds=vald_preds > 0,
                        class_names=["Healthy","Sick"])})
                    torch.save({
                        'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_staגte_dict': self.optimizers.state_dict(),
                            'loss': loss,
                            'accuracy':accuracy,
                            'f1':f1,
                            'seed':self.seed
                            }, f"{self.hp['checkpoints']}/{self.hp['name']}/f1_based_model.pt")

                max_validation_f1 = max(max_validation_f1,f1)
                max_validation_acc = max(max_validation_acc,accuracy)
                wandb.run.summary["validation_accuracy.max"] = max_validation_acc

                if (accuracy.item()>last_acc):
                    last_acc=accuracy.item()
                    runs_without_improv=0
                else:
                    runs_without_improv+=1

                pbar_epochs.set_description(f"validation epoch {epoch}, train acc {'{:.2f}'.format(epoch_accuracy.item())}, validation acc {'{:.2f}'.format(accuracy.item())}")            

                if (runs_without_improv>=self.early_stop):
                    vald_losses += [vald_loss]
                    return self.model,train_losses,vald_loss, train_accuracies,train_precisions,train_recalls,vald_accuracies,vald_precisions,vald_recalls                


                vald_losses += [vald_loss]

        return self.model,train_losses,vald_loss, train_accuracies,train_precisions,train_recalls,vald_accuracies,vald_precisions,vald_recalls

    def test(self,model):
        test_loss = 0.0
        test_accuracies = []
        test_precisions = []
        test_recalls = []
        test_f1 = []
        dict_pathologies = {}
        with tqdm(self.test_set) as t:
            for idx,sample in enumerate(t):
                x = sample['data'].to(device=self.device)
                y = sample['classification'].float().squeeze().to(device=self.device)
                
                for original_class_entry in sample['original_class']:
                    if original_class_entry not in list(dict_pathologies.keys()):
                        dict_pathologies[original_class_entry]={'Correct':0,'Incorrect':0}
                
                with torch.no_grad():
                    outputs = model(x)
                    loss = self.critereon(outputs,y)
                    test_loss += loss.item() 
                    predictions = outputs.detach() > 0
                    
                    correct_decisions = predictions == y
                    for (pathology,decision) in zip(sample['original_class'],correct_decisions):
                        dict_pathologies[pathology]['Correct'] += 1 if decision else 0
                        dict_pathologies[pathology]['Incorrect'] += 1 if not decision else 0

                    len_predictions = 1 if len(predictions.shape) == 0 else len(predictions)
                    accuracy = torch.sum(predictions==y)/len_predictions
                    precision = torch.sum(predictions*(predictions==y))/torch.sum(predictions)
                    recall = torch.sum(predictions*(predictions==y))/torch.sum(y)
                    f1 = f1_score(y.cpu(),predictions.cpu())

                    test_accuracies += [accuracy]
                    test_precisions += [precision]
                    test_recalls += [recall]
                    test_f1 += [f1]
                    self.writer.add_scalar('Loss/test',loss.item(),idx)
                    self.writer.add_scalar('Accuracy/test',accuracy,idx)
                    self.writer.add_scalar('Precision/test',precision,idx)
                    self.writer.add_scalar('Recall/test',recall,idx)
                    self.writer.add_scalar('Recall/test',f1,idx)

            test_accuracies = sum(test_accuracies)/len(test_accuracies)
            test_precision = sum(test_precisions)/len(test_precisions)
            test_recall = sum(test_recalls)/len(test_recalls)
            test_f1 = sum(test_f1)/len(test_f1)

        print(f"final acc {test_accuracies*100}%  final f1 {test_f1} final precision {test_precision} final recall {test_recall}")
        # print(dict_pathologies)
        # print(f"len of dict {len(dict_pathologies.keys())}")
        data = list(dict_pathologies.items())
        # print(data[0])
        keys = [entry[0] for entry in data]
        data = [[entry[1]['Correct'],entry[1]['Incorrect']] for entry in data]
        df_cm = pd.DataFrame(data,keys,['Correct','Incorrect'])
        sn.set(font_scale=0.8) # for label size
        sn.set(rc={'figure.figsize':(15,15)})

        sn.heatmap(df_cm, annot=True,fmt='d' ,annot_kws={"size": 16}) # font size

        plt.savefig('confusion_matrix.png')
        return test_loss,test_accuracies,test_precision,test_recall
