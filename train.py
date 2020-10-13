# TODO comments

import numpy as np
from utils import parse_filename
from data import build_data_from_arff


def train(train_data, n_hidden=None, learn_rate=0.1, n_epochs=500):
    n_in, n_out = len(train_data[0][0]), len(train_data[0][1])

    # default: set size of hidden layer to average of input and output layer
    if n_hidden is None:
        n_hidden = (n_in + n_out) // 2

    # initialize weights to small random values
    w_hidden = _init_weights(n_hidden, n_in)
    w_out = _init_weights(n_out, n_hidden)

    for epoch in range(n_epochs):
        for input, target in train_data:
            hidden, outputs = forward(input, w_hidden, w_out)
            partial_outs, partial_hiddens = _backprop(hidden, outputs, target, w_out)

            # update weights
            # w_hidden += learn_rate * (partial_hiddens[:, None] * input)
            # w_out += learn_rate * (partial_outs[:, None] * hidden)
            for j in range(w_hidden.shape[0]):
                for i in range(w_hidden.shape[1]):
                    w_hidden[j][i] += learn_rate * partial_hiddens[j] * input[i]

            for j in range(w_out.shape[0]):
                for i in range(w_out.shape[1]):
                    w_out[j][i] += learn_rate * partial_outs[j] * hidden[i]

    return w_hidden, w_out


def forward(input, w_hidden, w_out):
    # inputs is Nx1
    # w_hidden is HxN
    # w_out is KxH

    hidden = _sigmoid(w_hidden @ input)   # H x 1
    outputs = _sigmoid(w_out @ hidden)    # K x 1

    return hidden, outputs


""" PRIVATE METHODS """


def _backprop(hidden, outputs, target, w_out):
    """
    Returns partial errors for hidden and output layer, via backpropagation
    rule. Assumes sigmoid activation follows both layers.

    See Mitchell 4.5.3 for derivation.

    INPUTS:
    - hidden: node values in hidden layer (shape: 1 x H)
    - outputs: node values in output layer (shape: 1 x K)
    - target: boolean feature vector for single training example
    - w_out: weights connecting hidden and output layer (shape: K x H)

    OUTPUTS:
    - partial_outs (numpy array): partial errors for output layer
    - partial_hiddens (numpy array): partial errors for hidden layer
    """

    # backprop partial errors for output layer (shape: 1 x K)
    partial_outs = outputs * (1 - outputs) * (target - outputs)

    # backprop partial errors for hidden layer (shape: 1 x H)
    partial_hiddens = []
    for h in range(len(hidden)):
        partial = hidden[h] * (1 - hidden[h])
        s = 0
        for k in range(len(outputs)):
            s += partial_outs[k] * w_out[k][h]
        partial *= s
        partial_hiddens.append(partial)

    partial_hiddens = np.array(partial_hiddens)

    # partial_hiddens = hidden * (1 - hidden) * (partial_outs @ w_out)

    return partial_outs, partial_hiddens


def _init_weights(n_rows, n_cols, std_dev=0.1):
    """
    Returns normally distributed weights (mean=0, sd=std_dev) with shape
    (n_rows, n_cols)
    """
    weights = np.random.randn(n_rows, n_cols) * std_dev
    return weights


def _sigmoid(x):
    """
    Returns sigmoid output given number or array-like of numbers.
    """
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    filename = parse_filename()
    data = build_data_from_arff(filename)
    for d in data:
        print(d)

    w_hidden, w_out = _backprop(data, 3, n_epochs=2)
    print("****")
    print(w_hidden)
    print(w_out)
