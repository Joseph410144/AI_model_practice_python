import numpy as np

def Sigmoid(x):
    return 1/(1+np.exp(x*-1))
weights = np.random.rand(5)
input_ = np.random.rand(5)


print(Sigmoid(sum(weights*input_)))