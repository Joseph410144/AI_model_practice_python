from Perceptron import Perceptron
from Adaline import Adaline
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
            Labeldata.append(0)
        elif data[-1].split()[0] == "Iris-versicolor":
            Labeldata.append(1)

    f.close

    return np.array(Traindata), np.array(Labeldata)
    
def Accuracy(y_label, y_output):
    ans = list(y_label == y_output)
    Acc = ans.count(True)/len(ans)

    return round(Acc, 2)*100

def main() -> None:
    # model = Perceptron(weight_num=4)
    model = Adaline(weight_num=4)
    lr = 0.001
    Datapath = r"Dataset\Iris\irisdata.txt"
    train_data, label_data = Dataset(Datapath)
    model.fit(epochs=1000, lr=lr, data=train_data, label=label_data)
    model.PlotLossStep()
    output = model.predict(train_data)

    print(f"Model Accuracy: {Accuracy(label_data, output)}%")



if __name__  == "__main__":
    main()