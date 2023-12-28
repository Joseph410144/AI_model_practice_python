from GeneralModel import NoEqualLenerror, GeneralModel
import numpy as np
import random

class Perceptron(GeneralModel):
    def __init__(self, weight_num) -> None:
        super().__init__()
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
        
        return self.ThresholdFun(sum(x*self.weights))
    
    def UpdateModelWeight(self, data, loss, lr) -> None:
        for weight_pos in range(self.weight_num):
            self.weights[weight_pos] += lr*loss*data[weight_pos]
            
    def ThresholdFun(self, Output) -> int:
        if Output > 0:
            return 1
        else:
            return 0
        



