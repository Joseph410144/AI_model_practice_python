from GeneralModel import NoEqualLenerror, GeneralModel
import numpy as np

class MutilayerPerceptron(GeneralModel):
    def __init__(self, input_num:int, weight_num:int) -> None:
        super().__init__()
        # Input layer
        self.input_num = input_num
        self.weight_num = weight_num
        """ There is only one bias in each layer  """
        self.Inputlayer_weight = np.random.rand((self.input_num+1)*weight_num)
        self.Inputlayer_weight_gradient = np.random.rand((self.input_num+1)*weight_num)


        # Output layer
        self.Outputlayer_weight = np.random.rand(self.weight_num+1)
        self.Outputlayer_weight_gradient = np.random.rand(self.weight_num+1)
    
    def forward(self, x) -> float:
        """ x is input"""
        x = np.array(x)
        x = np.append(x, 1)
        """ 
        Input Layer
        Input Shape: (input_num+1, ) 
        Output Shape: (weight_num, ) >> bias is always is 1
        """
        self.input_layer_output = []
        for i in range(self.weight_num):
            self.input_layer_output.append(np.dot(x, self.Inputlayer_weight[i*len(x):(i*len(x))+len(x)]))

        self.input_layer_output = np.array(self.input_layer_output)

        """
        Output Layer
        Input Shape: (weight_num+1, )
        Output Shape: (1, )
        """
        Output_layer_input = self.input_layer_output
        Output_layer_input = np.append(Output_layer_input, 1)
        Output = np.dot(Output_layer_input, self.Outputlayer_weight)

        # del Output_layer_input
        return self.Sigmoid(Output)
    
    def predict(self, TestData) -> np.ndarray:
        ans = []
        for batch in range(TestData.shape[0]):
            model_output = self.ThresholdFun(self.forward(TestData[batch]))
            ans.append(model_output)
        
        return np.array(ans)
    
    def Sigmoid(self, x) -> float:
        return 1/(1+np.exp(x*-1))
    
    def gradient_delta(self, y_true, y_output) -> float:
        return (y_output-y_true)*y_output*(1-y_output)
    
    def get_weight_gardient(self, y_true, y_output, x) -> None:
        """ Output Layer """
        for i in range(len(self.Outputlayer_weight_gradient)-1):
            self.Outputlayer_weight_gradient[i] = self.gradient_delta(y_true, y_output)*self.input_layer_output[i]
        self.Outputlayer_weight_gradient[-1] = self.gradient_delta(y_true, y_output)*1

        """ Input Layer """
        for i in range(len(self.Inputlayer_weight_gradient)):
            if i%(self.input_num+1) == self.input_num:
                x_gra = 1
            else:
                x_gra = x[i%(self.input_num+1)]
            
            if (i-self.weight_num) < 0:
                Out_gra = self.Outputlayer_weight[0]
            else:
                Out_gra = self.Outputlayer_weight[(i-self.weight_num)//self.weight_num+1]

            self.Inputlayer_weight_gradient[i] = self.gradient_delta(y_true, y_output)*x_gra*Out_gra

    def UpdateModelWeight(self, data, loss, lr, y_output, y_true) -> None:
        self.get_weight_gardient(y_true, y_output, data)
        for i in range(len(self.Outputlayer_weight_gradient)):
            self.Outputlayer_weight[i] -= self.Outputlayer_weight_gradient[i]*lr

        for i in range(len(self.Inputlayer_weight_gradient)):
            self.Inputlayer_weight[i] -= self.Inputlayer_weight_gradient[i]*lr

    def loss(self, y_true, y_output) -> float:
        return 1/2*(y_true - y_output)**2
    
    def ThresholdFun(self, Output) -> int:
        if Output > 0.5:
            return 1
        else:
            return 0

if __name__ == "__main__":
    Mlp = MutilayerPerceptron(input_num=4, layer_num=1, weight_num=5)
    print(Mlp.Inputlayer_weight.shape, Mlp.Hiddenlayer_weight.shape, Mlp.Outputlayer_weight.shape)
    print(Mlp.forward([1, 2, 3, 4]))