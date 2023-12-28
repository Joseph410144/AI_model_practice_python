import numpy as np

class VectorSubtraction_Loss():
    def __init__(self) -> None:
        pass

    def loss(self, y_output, y_true):
        return y_true - y_output