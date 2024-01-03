# AI_model_practice_python

GeneralModel
===============
Prototype for all model
Include Basic Method
1. fit
2. PlotLossStep

Perceptron
===============
Binary Classification
Activation function: step function

Adaline
===============
Binary Classification
Activation function: Sigmoid function

Gradient Descent
---------------
 $let\ loss\ function:$ $loss(y\_true, y\_output)={1\over2}(y\_true-y\_output)^2,y\_true\ is\ label\ and\ y\_output=Sigmoid(x)$

 $and\ gradient\ descent\ is\ W_{i+1}=W_i-\eta\nabla loss(W_i), Sigmoid\ function\ is\ {1\over {1+e^{-x}}},where\ x=W^\mathrm{T}X$

 $also\ by\ chain\ rule: $
 $1. {\mathrm{d}loss\over \mathrm{d}w_i}={{\partial loss \over \partial y\_output} {\mathrm{d}y\_output \over \mathrm{d}w_i}}+{{\partial loss \over \partial y\_true} {\mathrm{d}y\_true \over \mathrm{d}w_i}} ={{\partial loss \over \partial y\_output} {\mathrm{d}y\_output \over \mathrm{d}w_i}}$

 $2. {\mathrm{d}y\_output \over \mathrm{d}w_i}={{\partial y\_output \over \partial x} {\mathrm{d}x \over \mathrm{d}w_i}}$

 $let\ y\_true=\hat y,\ y\_output=y$

 $we\ can\ derive\ that$

 ${\partial loss \over \partial y}={1\over2}(2y-2\hat y)=y-\hat y$

 ${\partial y \over \partial x}={\partial Sigmoid(x) \over \partial x}={1\over {1+e^{-x}}}(1-{1\over {1+e^{-x}}})$ 

 ${\mathrm{d}x \over \mathrm{d}w_i}={\mathrm{d}(w_1x_1+....+w_nx_n) \over \mathrm{d}w_i}=x_i$

 ${\mathrm{d}loss\over \mathrm{d}w_i}={{\partial loss \over \partial y} {{\partial y \over \partial x} {\mathrm{d}x \over \mathrm{d}w_i}}}=x_i(y-\hat y)\times{1\over {1+e^{-x}}}(1-{1\over {1+e^{-x}}})$


