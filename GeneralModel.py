import numpy as np
import random
import matplotlib.pyplot as plt

class NoEqualLenerror(Exception):
    def __init__(self, message="input data length is not equal weight numbers"):
        self.message = message
        super().__init__(self.message)

class TrainingModel():
    def __init__(self, epochs, lr, data, label, criterion) -> None:
        self.epochs = epochs
        self.lr = lr
        self.data = data # shape(data_numbers, data vector)
        self.label = label # shape(data_numbers, 1)
        self.criterion = criterion
        self.loss_fig = []

    def fit(self, model) -> None:
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in range(self.data.shape[0]):
                model_output = model.forward(self.data[batch])
                loss = self.criterion.loss(model_output, self.label[batch])
                
                self.UpdateModelWeight(model, self.data[batch], loss, self.lr)
                epoch_loss += loss

            epoch_loss /= self.data.shape[0]
            self.loss_fig.append(abs(epoch_loss))
            print(f"Epoch {epoch}| loss = {abs(epoch_loss)}")
        
    def UpdateModelWeight(self, model, data, loss, lr) -> None:
        for weight_pos in range(model.weight_num):
            model.weights[weight_pos] += lr*loss*data[weight_pos]
    
    def PlotLossStep(self):
        plt.plot(self.loss_fig)
        plt.title("Loss step")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.show()

class GeneralModel():
    def __init__(self) -> None:
        pass

    def fit(self, epochs=20, lr=0.001, data=None, label=None) -> None:
        self.epochs = epochs
        self.lr = lr
        self.data = data # shape(data_numbers, data vector)
        self.label = label # shape(data_numbers, 1)
        self.loss_fig = []

        """ Training iteration """
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in range(self.data.shape[0]):
                model_output = self.forward(self.data[batch])
                loss = self.loss(self.label[batch], model_output)
                
                self.UpdateModelWeight(self.data[batch], loss, self.lr, model_output, self.label[batch])
                epoch_loss += loss

            epoch_loss /= self.data.shape[0]
            self.loss_fig.append(abs(epoch_loss))
            print(f"Epoch {epoch}| loss = {abs(epoch_loss)}")
    
    def UpdateModelWeight(self, data, loss, lr, y_output, y_true) -> None:
        pass
    
    def forward(self) -> None:
        pass
    
    def loss(self) -> None:
        pass

    def PlotLossStep(self):
        plt.plot(self.loss_fig)
        plt.title("Loss step")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.show()