from pickletools import optimize
import torch
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self,dataset,model,optimizers,critereon,hyper_params) -> None:
        # self.dl = dataloader
        self.train_set, self.val_set, self.test_set = self.train_val_test_split(dataset)
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
       
        self.model = model
        self.optimizers = optimizers
        self.critereon=critereon
        self.hp = hyper_params
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device=self.device)
    def train_val_test_split(self,ds):
        ds_len =  len(ds)
        len_test = math.floor(ds_len*0.2)
        len_train_vald = ds_len - len_test
        len_train = math.ceil(len_train_vald*0.8)
        len_vald = len_train_vald - len_train
        return torch.utils.data.random_split(ds, [len_train,len_vald,len_test]) 

    def train(self):
        train_losses = []
        vald_losses = []

        train_accuracies = []
        train_precisions = []
        train_recalls = []
        
        for epoch in range(self.hp['epochs']):
            running_loss = 0.0
            epoch_accuracies = []
            epoch_precisions = []
            epoch_recalls = []
            with tqdm(self.train_set) as pbar:
                for idx,sample in enumerate(pbar):                
                    x = sample['data'].to(device=self.device)
                    y = sample['classification'].float().squeeze().to(device=self.device)
                    self.optimizers.zero_grad()
                    outputs = self.model(x)
                    loss = self.critereon(outputs,y)
                    loss.backward()
                    self.optimizers.step()
                    running_loss += loss.item() 
                    predictions = outputs.detach() > 0.5
                    len_predictions = 1 if len(predictions.shape) == 0 else len(predictions)
                    accuracy = torch.sum(predictions==y)/len_predictions      
                    precision = torch.sum(predictions*(predictions==y))/torch.sum(predictions)
                    recall = torch.sum(predictions*(predictions==y))/torch.sum(y)

                    train_accuracies += [accuracy]
                    train_precisions += [precision]
                    train_recalls += [recall]
                    
                    epoch_accuracies += [accuracy]
                    epoch_precisions += [precision]
                    epoch_recalls += [recall]

                    pbar.set_description(f"train epoch {epoch}, train loss is {loss.item()}, Accuracy {accuracy*100}%, Precision {precision*100}%, Recall {recall*100}%")
                    if idx == len(self.train_set)-1 :
                        epoch_accuracy = sum(epoch_accuracies)/len(epoch_accuracies)
                        epoch_precision = sum(epoch_precisions)/len(epoch_precisions)
                        epoch_recall = sum(epoch_recalls)/len(epoch_recalls)
                        pbar.set_description(f"train epoch {epoch}, train loss:{running_loss} , Mean Accuracy:{epoch_accuracy*100}%, Mean Precision:{epoch_precision*100}%, Mean Recall:{epoch_recall*100}%")

                    self.writer.add_scalar('Loss/train',loss.item(),idx)
                    self.writer.add_scalar('Accuracy/train',accuracy,idx)
                    self.writer.add_scalar('Precision/train',precision,idx)
                    self.writer.add_scalar('Recall/train',recall,idx)

            train_losses += [running_loss]
            vald_accuracies = []
            vald_precisions = []
            vald_recalls = []

            vald_loss = 0.0
            with tqdm(self.val_set) as t:
                    for idx,sample in enumerate(t):
                        x = sample['data'].to(device=self.device)
                        y = sample['classification'].float().squeeze().to(device=self.device)
                        with torch.no_grad():
                            outputs = self.model(x)
                            loss = self.critereon(outputs,y)
                            vald_loss += loss.item() 
                            predictions = outputs.detach() > 0.5
                            len_predictions = 1 if len(predictions.shape) == 0 else len(predictions)
                            accuracy = torch.sum(predictions==y)/len_predictions             
                            precision = torch.sum(predictions*(predictions==y))/torch.sum(predictions)
                            recall = torch.sum(predictions*(predictions==y))/torch.sum(y)
                            vald_accuracies += [accuracy]
                            vald_precisions += [precision]
                            vald_recalls += [recall]
                        self.writer.add_scalar('Loss/validation',loss.item(),idx)
                        self.writer.add_scalar('Accuracy/validation',accuracy,idx)
                        self.writer.add_scalar('Precision/validation',precision,idx)
                        self.writer.add_scalar('Recall/validation',recall,idx)
                        t.set_description(f"validation epoch {epoch}, validation loss is {loss.item()}, Accuracy {accuracy*100}%, Precision {precision*100}%, Recall {recall*100}%")
            
            vald_losses += [vald_loss]

        return self.model,train_losses,vald_loss, train_accuracies,train_precisions,train_recalls,vald_accuracies,vald_precisions,vald_recalls

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
                        predictions = outputs.detach() > 0.5
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
