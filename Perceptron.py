from GeneralModel import NoEqualLenerror, GeneralModel
import numpy as np
import random

class Perceptron(GeneralModel):
    def __init__(self, weight_num) -> None:
        super().__init__()
        self.weight_num = weight_num
        self.weights = np.random.rand(self.weight_num+1)
        
    def forward(self, x) -> int:
        """ x is input"""
        x = np.array(x)
        x = np.append(x, 1)
        if len(x) != len(self.weights):
            raise NoEqualLenerror()
        
        return self.ThresholdFun(sum(x*self.weights))
    
    def predict(self, TestData) -> np.ndarray:
        ans = []
        for batch in range(TestData.shape[0]):
            model_output = self.forward(TestData[batch])
            ans.append(int(model_output))
        
        return np.array(ans)
    
    def UpdateModelWeight(self, data, loss, lr, y_output, y_true) -> None:
        for weight_pos in range(self.weight_num):
            self.weights[weight_pos] += lr*loss*data[weight_pos]
    
    def loss(self, y_true, y_output) -> float:
        return y_true - y_output
            
    def ThresholdFun(self, Output) -> int:
        if Output > 0:
            return 1
        else:
            return 0
        



