from GeneralModel import NoEqualLenerror, GeneralModel
import numpy as np
import random

class Adaline(GeneralModel):
    def __init__(self, weight_num) -> None:
        super().__init__()
        self.weight_num = weight_num
        self.weights = np.random.rand(self.weight_num+1)
        
    def forward(self, x) -> float:
        """ x is input"""
        x = np.array(x)
        x = np.append(x, 1)
        if len(x) != len(self.weights):
            raise NoEqualLenerror()
        
        return self.Sigmoid(sum(x*self.weights))
    
    def predict(self, TestData) -> np.ndarray:
        ans = []
        for batch in range(TestData.shape[0]):
            model_output = self.ThresholdFun(self.forward(TestData[batch]))
            ans.append(model_output)
        
        return np.array(ans)

    def UpdateModelWeight(self, data, loss, lr, y_output, y_true) -> None:
        for weight_pos in range(self.weight_num):
            gd = self.gradient(y_true, y_output, data[weight_pos])
            self.weights[weight_pos] -= lr*gd
        
        """ update bias """
        gd = self.gradient(y_true, y_output, 1)
        self.weights[-1] -= lr*gd

    
    def loss(self, y_true, y_output) -> float:
        return 1/2*(y_true - y_output)**2
        # return y_true - y_output
    
    def gradient(self, y_true, y_output, x) -> float:
        return x*(y_output-y_true)*y_output*(1-y_output)

    def Sigmoid(self, x) -> float:
        return 1/(1+np.exp(x*-1))

    def ThresholdFun(self, Output) -> int:
        if Output > 0.5:
            return 1
        else:
            return 0