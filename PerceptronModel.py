import numpy as np
import random
    
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

    def fit(self, model):
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in range(self.data.shape[0]):
                model_output = model.forward(self.data[batch])
                loss = self.criterion(model_output, self.label[batch])
                model = self.UpdateModelWeight(model, self.data[batch], loss, self.lr)
                total_loss += loss
        
        total_loss /= self.epochs

        return model, total_loss
                
    def UpdateModelWeight(self, model, data, loss, lr):
        for weight_pos in range(model.weight_num):
            model.weights[weight_pos] += lr*loss*data[weight_pos]
        
        return model

class PerceptronModel():
    def __init__(self, weight_num) -> None:
        self.weight_num = weight_num
        self.weights = np.zeros(self.weight_num+1) 
        for i in range(self.weight_num+1):
            self.weights[i] = random.random()
        
    def forward(self, x):
        """ x is input"""
        x = np.array(x)
        x = np.append(x, 1)
        if len(x) != len(self.weights):
            raise NoEqualLenerror()
        
        return self.ThresholdFun(x*self.weights)
            
    def ThresholdFun(self, Output) -> int:
        if Output > 0:
            return 1
        else:
            return 0

class Loss_function():
    def __init__(self) -> None:
        pass

    def loss(y_output, y_true):
        return abs(y_true - y_output)

