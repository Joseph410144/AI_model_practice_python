from Perceptron import Perceptron
from GeneralModel import TrainingModel
from LossFunction import VectorSubtraction_Loss
import matplotlib.pyplot as plt
import numpy as np

def Dataset(path):
    f = open(path)
    Traindata = []
    Labeldata = []
    for line in f.readlines():
        if str(line).split(" ")[0] == "\n":
            continue
        batchdata = []
        data = str(line).split(",")
        if data[-1].split()[0] == "Iris-virginica":
            continue
        for i in range(len(data)-1):
            batchdata.append(float(data[i]))
        Traindata.append(batchdata)
        if data[-1].split()[0] == "Iris-setosa":
            Labeldata.append(-1)
        elif data[-1].split()[0] == "Iris-versicolor":
            Labeldata.append(1)

    f.close

    return np.array(Traindata), np.array(Labeldata)
    
def main() -> None:
    model = Perceptron(weight_num=4)
    criterion = VectorSubtraction_Loss()
    lr = 0.0001
    Datapath = r"Dataset\Iris\irisdata.txt"
    train_data, label_data = Dataset(Datapath)
    model.fit(epochs=50, lr=lr, data=train_data, label=label_data, criterion=criterion)
    model.PlotLossStep()
    # train = TrainingModel(epochs=50, lr=lr, data=train_data, label=label_data, criterion=criterion)
    # train.fit(model)
    # train.PlotLossStep()


if __name__  == "__main__":
    main()