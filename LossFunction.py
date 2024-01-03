import numpy as np
import math

class VectorSubtraction_Loss():
    def __init__(self) -> None:
        pass

    def loss(self, y_output, y_true):
        return y_true - y_output

class AdalineLoss():
    def __init__(self) -> None:
        pass

    def loss(self, y_output, y_true):
        return (1/2)*(y_true - y_output)**2
    
