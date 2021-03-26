import sys
import copy
import torch
import threading
from tqdm import tqdm
from torch.utils.data import DataLoader


class Client:
    
    def __init__(self, idx, dataset, batch_size, optimizer, optim_params, scheduler,
                 scheduler_params, loss_fn, num_epochs):
        self.idx = idx
        self.dataset = dataset
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = optimizer
        self.optim_params = optim_params
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.loss_fn = loss_fn.to(self.device)
        self.num_epochs = num_epochs
        self.iter_per_epoch = len(self.dataloader)
        self.len_dataset = len(self.dataset)
        self.stop_flag = False
    
    def updateModel(self, model):
        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer_fn = self.optimizer(self.model.parameters(), **self.optim_params)
        self.optim_scheduler = self.scheduler(self.optimizer_fn, **self.scheduler_params)

    def __epoch(self):

        epoch_loss = 0
        tot_correct_predictions = 0

        # for images, targets in tqdm(self.dataloader, position=0, leave=True, file=sys.stdout):
        for images, targets in self.dataloader:
            
            if self.stop_flag:
                break

            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer_fn.zero_grad()
            output_probs = self.model(images)

            loss = self.loss_fn(output_probs, targets)
            loss.backward()
            self.optimizer_fn.step()
            
            predictions = torch.argmax(output_probs, dim=1)

            correct_predictions = torch.sum(predictions == targets).item()
            tot_correct_predictions += correct_predictions

            epoch_loss += loss.item()
            
        self.optim_scheduler.step()
        
        avg_loss = epoch_loss/self.iter_per_epoch
        accuracy = tot_correct_predictions/self.len_dataset*100

        return avg_loss, accuracy

    def train(self):

        self.model.train()

        # print(f"CLIENT ID={self.idx}\n\n")

        # for epoch in range(self.num_epochs):
        #     print(f"EPOCH {epoch+1}/{self.num_epochs}\n")
        #     avg_loss, train_accuracy = self.__epoch()
        #     print(f"Train: Loss={round(avg_loss, 3)}, Accuracy={round(train_accuracy, 2)}%\n")

        for epoch in range(self.num_epochs):
            if self.stop_flag:
                break
            print(f"tid={str(threading.get_ident())[-7:]} - k_id={self.idx}: START EPOCH={epoch+1}/{self.num_epochs}")
            avg_loss, train_accuracy = self.__epoch()
            print(f"tid={str(threading.get_ident())[-7:]} - k_id={self.idx}: END   EPOCH={epoch+1}/{self.num_epochs} - ", end="")
            print(f"Loss={round(avg_loss, 3)}, Accuracy={round(train_accuracy, 2)}%")
        
        return self.model.state_dict()

    def __len__(self):
        return len(self.dataset)