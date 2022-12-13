from trainer.BaseTrainer import BaseTrainer,BatchResult
import torch
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt
import seaborn as sn
import core
import pandas as pd
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_curve, PrecisionRecallDisplay, average_precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
# from math import prod

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm * 100

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(16, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)  


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    return plt


class MultiClassSingleLabelTrainer(BaseTrainer):
    def __init__(self, datasets, model, optimizer, hyper_params, early_stop=..., device=None, verbose=False, logResults=True, confusion_mat_every=5, classes_amount=10, criterion = None) -> None:
        super().__init__(datasets, model, optimizer, hyper_params, early_stop, device, verbose, logResults, confusion_mat_every)
        self.classes_amount = classes_amount      
        if criterion == None:
            self.criterion = nn.CrossEntropyLoss()   
        else:
            self.criterion = criterion   

    def train_batch(self, sample) -> BatchResult:        
        x = sample['data'].to(device=self.device)
        y = sample['classification'].squeeze().to(device=self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x)        
        loss = self.criterion(outputs,y)
        loss.backward()
        self.optimizer.step()
        
        predictions = torch.argmax(outputs, dim=1)

        len_predictions = torch.numel(y)
        accuracy = torch.sum(predictions==y)/len_predictions      
        return BatchResult(predictions.cpu().detach().numpy(),outputs.cpu().detach().numpy(),loss,accuracy)

    def validate_batch(self, sample) -> BatchResult:
        x = sample['data'].to(device=self.device)
        y = sample['classification'].squeeze().to(device=self.device)
        accuracy = 0.0
        with torch.no_grad():
            outputs = self.model(x)
            loss = self.criterion(outputs,y)            
            predictions = torch.argmax(outputs, dim=1)
            len_predictions = torch.numel(y)
            accuracy = torch.sum(predictions==y)/len_predictions             
            return BatchResult(predictions.cpu().detach().numpy(),outputs.cpu().detach().numpy(),loss,accuracy)

    def process_valid_results(self, valid_pred, valid_scores, valid_gt, epoch):
        classes = list(core.params.PathologiesToIndex.keys())

        cf_mat = confusion_matrix(valid_gt, valid_pred)        
        # fig = ConfusionMatrixDisplay(cf_mat,classes).plot().figure_  
        plt = plot_confusion_matrix(cf_mat, classes[1:])      
        os.makedirs(core.params.results_dir + f'/singleLabel_confusion_matrices',exist_ok=True)
        plt.savefig(core.params.results_dir + f'/singleLabel_confusion_matrices/output_{epoch}.png')

    def process_test_results(self, valid_pred, valid_scores, valid_gt, epoch):
        classes = list(core.params.PathologiesToIndex.keys())

        cf_mat = confusion_matrix(valid_gt, valid_pred)        
        # fig = ConfusionMatrixDisplay(cf_mat,classes).plot().figure_  
        plt = plot_confusion_matrix(cf_mat, classes[1:])      
        os.makedirs(core.params.results_dir + f'/singleLabel_confusion_matrices',exist_ok=True)
        plt.savefig(core.params.results_dir + f'/singleLabel_confusion_matrices/output_test_{epoch}.png')


    

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
            cf = cf_matrix[c]
            summed = cf.sum(axis=1)
            gt_not_sick = summed[0]
            gt_sick = summed[1]
            cf = cf + 0.0
            cf_pre = np.zeros_like(cf)
            cf_pre[0,:] = (cf[0,:]/summed[0])*100
            cf_pre[1,:] = (cf[1,:]/summed[1])*100
            cf_pre = cf_pre.round(2)

            df_cm = pd.DataFrame(cf_pre, index = [f"Not Sick GT ({gt_not_sick})",f"Sick GT ({gt_sick})"],
                                columns = ["Not Sick Pred","Sick Pred"])
            plt.figure(figsize = (10,7))
            plt.rc('font', size=20)
            sn.heatmap(df_cm, annot=True,fmt='g')
            os.makedirs(core.params.results_dir + f'/confusion_matrices',exist_ok=True)
            plt.savefig(core.params.results_dir + f'/confusion_matrices/output_{epoch}_{classes[c]}.png')
            plt.close('all') 

    def log_precision_recall(self, valid_scores, valid_gt, epoch):      
        valid_proba_t = nn.functional.sigmoid(torch.Tensor(valid_scores))
        valid_gt_t = np.array(valid_gt)
        classes = list(core.params.PathologiesToIndex.keys())
        for i in range(self.classes_amount):
            prec, rec, thresholds = precision_recall_curve(valid_gt_t[:,i], valid_proba_t[:,i])
            avg_prec = average_precision_score(valid_gt_t[:,i], valid_proba_t[:,i])         

            os.makedirs(core.params.results_dir + f'/precision_recalls',exist_ok=True)
            display = PrecisionRecallDisplay(recall=rec, precision=prec, average_precision=avg_prec)
            display.plot().figure_.savefig(core.params.results_dir + f'/precision_recalls/output_{epoch}_{classes[i]}.png')  

            os.makedirs(core.params.results_dir + f'/average_precision_logs',exist_ok=True)
            file_mode = 'a'
            if epoch == 0:
                file_mode = 'w+'
            
            f = open(core.params.results_dir + f'/average_precision_logs/{classes[i]}_log.txt',file_mode)
            f.write(f'epoch_{epoch}: {str(avg_prec)}\n')
            f.close()

    def log_f1_scores(self, valid_pred, valid_gt, epoch):
        f1_scores = f1_score(valid_gt, valid_pred, average=None)        

        classes = list(core.params.PathologiesToIndex.keys())
        for i in range(self.classes_amount):            
            os.makedirs(core.params.results_dir + f'/f1_scores',exist_ok=True)            
            file_mode = 'a'
            if epoch == 0:
                file_mode = 'w+'
            
            with open(core.params.results_dir + f'/f1_scores/{classes[i]}_log.txt',file_mode) as f:
                f.write(f'epoch_{epoch}: {str(f1_scores[i])}\n')

    def log_precsion_recall_fscore(self, valid_pred, valid_gt, epoch):
        classes = list(core.params.PathologiesToIndex.keys())
        os.makedirs(core.params.results_dir + f'/precision_recall_fscores',exist_ok=True)

        valid_pred_t = np.array(valid_pred)
        valid_gt_t = np.array(valid_gt)        

        for i in range(self.classes_amount):
            prec, rec, fscore, _ = precision_recall_fscore_support(valid_gt_t[:,i], valid_pred_t[:,i],average='binary')  

            file_mode = 'a'
            if epoch == 0:
                file_mode = 'w+'
            
            with open(core.params.results_dir + f'/precision_recall_fscores/{classes[i]}_log.txt',file_mode) as f:                
                writer = csv.writer(f)
                writer.writerow([prec,rec,fscore])


    def process_valid_results(self, valid_pred, valid_scores, valid_gt, epoch):        
        self.log_precsion_recall_fscore(valid_pred, valid_gt, epoch)        

        if epoch % self.confusion_mat_every == 0:
            self.log_confusion_matrix(valid_pred, valid_gt, epoch)
