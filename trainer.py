import torch

import numpy as np

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

        y_train = []
        for _, labels in self.train_dataloader:
            y_train.extend(labels.tolist())

        y_train = np.array(y_train)

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size = VAL_SIZE,
            shuffle = True
        )

        y_val = []
        for _, labels in self.val_dataloader:
            y_val.extend(labels.tolist())

        y_val = np.array(y_val)
        
        _, axes = plt.subplots(1, 2)
        axes[0].hist(y_train, bins = 10)
        axes[1].hist(y_val, bins = 10)

        plt.show()

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
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters: ", num_params)

        num_batches = num_batches = math.ceil(len(self.train_dataset) / BATCH_SIZE)

        print()
        print("Start training: ...")

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

                        print("Start Epoch {}/{}: Train accuracy = {}%, Val accuracy = {}%".format(epoch_no, num_epochs, round(train_accuracy * 100, 2), round(val_accuracy * 100, 2)))
                    
                    self.model.train()

                if batch_no % 200 == 0:
                    print("- At Batch no {}/{}: Train loss = {}".format(batch_no, num_batches, loss))  

            print()

            if epoch_no % 99 == 0:       
                self.__report()

            if epoch_no % 100 == 0:
                choice = input("End training? (Y/N): ")
                if choice == "Y":
                    break
            
        choice = input("Finish training! Save model? (Y/N): ")
        if choice == "Y":
            torch.save(self.model.state_dict(), MODEL_STATE_DICT_FILE)
