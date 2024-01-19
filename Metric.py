import numpy as np

def Accuracy(y_label, y_output):
    ans = list(y_label == y_output)
    Acc = ans.count(True)/len(ans)

    return round(Acc, 2)*100