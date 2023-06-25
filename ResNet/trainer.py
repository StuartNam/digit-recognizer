import torch

import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

import math

from config import *

class Trainer:
    def __init__(
            self, 
            model, 
            dataset, 
            loss_fn, 
            optimizer, 
            auto_training_enabled = True, 
            num_epochs_to_check_end_training = None, 
            early_stopping_enabled = True, 
            max_patience = 10, 
            num_epochs_to_report = 25
        ):
        self.train_dataset, self.val_dataset = random_split(dataset, [TRAIN_SIZE, VAL_SIZE])
        self.train_dataloader = DataLoader(
            dataset = self.train_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True
        )
        
        self.val_dataloader = DataLoader(
            dataset = self.val_dataset,
            batch_size = len(self.val_dataset),
            shuffle = True
        )

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.best_state = None

        # Auto training
        self.auto_training_enabled = auto_training_enabled
        self.num_epochs_to_check_end_training = num_epochs_to_check_end_training
        # Report
        self.train_loss_over_time = []
        self.val_loss_over_time = []
        self.train_accuracy_over_time = []
        self.val_accuracy_over_time = []

        self.num_epochs_to_report = num_epochs_to_report

        # Early stopping support
        self.best_val_loss = torch.inf
        self.max_patience = max_patience
        self.early_stopping_enabled = early_stopping_enabled

    def __save_model(self, model_state_dict):
        self.best_state = model_state_dict

    def __report(self):
        # Plot train and val loss and accuracy
        _, axes = plt.subplots(1, 2)
        axes[0].plot(self.train_loss_over_time)
        axes[1].plot(self.val_loss_over_time)
        axes[1].plot(self.train_accuracy_over_time)
        axes[1].plot(self.val_accuracy_over_time)

        plt.show()

        # Additional report here
        # ...

        # Reset
        self.train_loss_over_time = []
        self.val_loss_over_time = []
        self.train_accuracy_over_time = []
        self.val_accuracy_over_time = []

    def __compute_train_val_loss(self, train_x, train_y, val_x, val_y):
        train_y_predicted = self.model(train_x)
        val_y_predicted = self.model(val_x)
        train_loss = self.loss_fn(
            train_y_predicted,
            train_y
        )

        val_loss = self.loss_fn(
            val_y_predicted,
            val_y
        )
        
        return train_loss, val_loss

    def train(self, num_epochs):
        num_params = sum(p.numel() for p in self.model.parameters())
        print("Number of model's parameters: ", num_params)

        num_batches = num_batches = math.ceil(len(self.train_dataset) / BATCH_SIZE)
        num_epochs_without_improvement = 0
        must_be_stopped_early = False

        print()
        print("Start training: ...")

        self.model.train()
        for epoch_no in range(num_epochs):
            for batch_no, (train_x, train_y) in enumerate(self.train_dataloader):
                if batch_no == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_iter = iter(self.val_dataloader)
                        val_x, val_y = next(val_iter)

                        train_loss, val_loss = self.__compute_train_val_loss(
                            train_x = train_x,
                            train_y = train_y,
                            val_x = val_x,
                            val_y = val_y
                        )

                        self.train_loss_over_time.append(train_loss.cpu())
                        self.val_loss_over_time.append(val_loss.cpu())
                        # self.train_accuracy_over_time.append(train_accuracy.cpu())
                        # self.val_accuracy_over_time.append(val_accuracy.cpu())

                        # Early stopping handler
                        if self.best_val_loss > val_loss:
                            self.best_val_loss = val_loss
                            self.__save_model(self.model.state_dict())

                            num_epochs_without_improvement = 0
                        else:
                            num_epochs_without_improvement += 1

                        if num_epochs_without_improvement == self.max_patience:
                            must_be_stopped_early = True
                        
                        print("Start Epoch {}/{}: Train loss = {}, Val loss = {}".format(epoch_no + 1, num_epochs, train_loss, val_loss))
                    
                    self.model.train()

                if must_be_stopped_early:
                    break

                train_y_predict = self.model(train_x)
                
                loss = self.loss_fn(train_y_predict, train_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_no != 0 and batch_no % 100 == 0:
                    print("- At Batch no {}/{}: Train loss = {}".format(batch_no + 1, num_batches, loss))

            print()

            if must_be_stopped_early:
                print("Early stopped at Epoch {}/{}!".format(epoch_no + 1, num_epochs))
                break

            if epoch_no != 0 and epoch_no % self.num_epochs_to_report == 0:       
                self.__report()

            if not self.auto_training_enabled:
                if epoch_no != 0 and epoch_no % self.num_epochs_to_check_end_training == 0:
                    choice = input("End training? (Y/N): ")
                    if choice == "Y":
                        break
        
        # Final report
        self.__report()

        print("Finish training!")
        torch.save(self.best_state, MODEL_FOLDER + TEMPORARY_FILE)

        choice = input("Save model? (Y/N): ")
        if choice == 'Y':
            torch.save(self.best_state, MODEL_FOLDER + MODEL_STATE_DICT_FILE)