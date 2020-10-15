# backprop-mlp

A simple implementation of backpropagation for a multilayer perceptron network with (at most) one hidden layer and discrete-valued inputs.

Completed as coursework for Williams College CSCI 374: Machine Learning.

## How to Run

The following command trains and evaluates the MLP net on the given `.arff`
file:

`python3 test.py -L 0.3 -E 300 -K 10 -H 5 NominalData/titanic.arff`

For this example, `-L 0.3` specifies a learning rate of 0.3, `-E 300` specifies
training for 300 epochs, `-K 10` specifies 10-fold cross validation, and `-H 5`
specifies the number of units in the hidden layer to be 5.

Running `python3 test.py -h` will also display the default values for these
parameters.
