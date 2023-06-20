import torch

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score

import math

from config import *

class Trainer:
    def __init__(self, model, dataset, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataset, self.val_dataset = random_split(dataset, [TRAIN_SIZE, VAL_SIZE])
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size = VAL_SIZE,
            shuffle = False
        )

        self.optimizer = optimizer

        self.train_loss_over_time = []
        self.val_loss_over_time = []
        self.train_accuracy_over_time = []
        self.val_accuracy_over_time = []

    def early_stopping():
        pass

    def __report(self):
        _, axes = plt.subplots(1, 2)
        axes[0].plot(self.train_loss_over_time)
        axes[0].plot(self.val_loss_over_time)
        axes[1].plot(self.train_accuracy_over_time)
        axes[1].plot(self.val_accuracy_over_time)

        plt.show()

    def train(self, num_epochs, lrate):
        num_batches = num_batches = math.ceil(len(self.train_dataset) / BATCH_SIZE)

        for epoch_no in range(num_epochs):
            for batch_no, (x, y) in enumerate(self.train_dataloader):
                y_predict = self.model(x)
                
                loss = self.loss_fn(y_predict, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_no == 0:
                    self.model.eval()
                    with torch.no_grad():
                        train_x, train_y = x, y
                        train_y_predict = self.model(train_x)
                        train_y_choice = torch.argmax(train_y_predict, dim = 1)
                        
                        train_loss = self.loss_fn(train_y_predict, train_y)
                        train_accuracy = accuracy_score(train_y.cpu(), train_y_choice.cpu())

                        val_iter = iter(self.val_dataloader)
                        val_x, val_y = next(val_iter)

                        val_y_predict = self.model(val_x)
                        val_y_choice = torch.argmax(val_y_predict, dim = 1)

                        val_loss = self.loss_fn(val_y_predict, val_y)
                        val_accuracy = accuracy_score(val_y.cpu(), val_y_choice.cpu())

                        self.train_loss_over_time.append(train_loss.cpu())
                        self.val_loss_over_time.append(val_loss.cpu())
                        self.train_accuracy_over_time.append(train_accuracy)
                        self.val_accuracy_over_time.append(val_accuracy)
                        print(train_loss, val_loss, train_accuracy, val_accuracy)
                    
                    self.model.train()

                if batch_no % 50 == 0:
                    print("Epoch {}/{}, Batch no {}/{}, Train loss = {}".format(epoch_no, num_epochs, batch_no, num_batches, loss))  

            if epoch_no % 20 == 0:       
                self.__report()

            if epoch_no % 20 == 0:
                choice = input("End training? (Y/N): ")
                if choice == "Y":
                    break

        choice = input("Finish training! Save model? (Y/N): ")
        if choice == "Y":
            torch.save(self.model.state_dict(), MODEL_STATE_DICT_FILE)
