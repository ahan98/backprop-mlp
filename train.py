# TODO comments

import numpy as np
from utils import parse_filename
from data import build_data_from_arff


def train(train_data, n_hidden=None, learn_rate=0.1, n_epochs=500):
    n_in, n_out = len(train_data[0][0]), len(train_data[0][1])

    if n_hidden is None:
        # set hidden layer size to aveage of input/output layer size
        n_hidden = (n_in + n_out) // 2

    # initialize weights to small random values
    w_hidden = _init_weights(n_hidden, n_in)
    w_out = _init_weights(n_out, n_hidden)

    for epoch in range(n_epochs):
        for input, target in train_data:
            # backprop weight partial errors
            hidden, outputs = forward(input, w_hidden, w_out)
            partial_outs, partial_hiddens = _backprop(hidden, outputs, target, w_out)

            # update weights
            w_hidden += learn_rate * (partial_hiddens[:, None] * input)
            w_out += learn_rate * (partial_outs[:, None] * hidden)

    return w_hidden, w_out


def forward(input, w_hidden, w_out):
    """
    Computes fully-connected feed-forward pass.

    Notation:
    N = total number of attribute values (including the bias weight)
    H = size of hidden layer
    K = size of output layer

    INPUTS:
    - input (ndarray): feature vector of training example (shape: (N,))
    - w_hidden (ndarray): weight matrix for hidden layer (shape: (H, N))
    - w_out(ndarray): weight matrix for output layer (shape: (K, N))

    OUTPUTS:
    - hidden (ndarray): node values in hidden layer (shape: (H,))
    - outputs (ndarray): node values in output layer (shape: (K,))
    """
    hidden = _sigmoid(w_hidden @ input)
    outputs = _sigmoid(w_out @ hidden)
    return hidden, outputs


""" PRIVATE METHODS """


def _backprop(hidden, outputs, target, w_out):
    """
    Returns partial errors for hidden and output layer, via backpropagation
    rule. Assumes sigmoid activation follows both layers.

    See Mitchell 4.5.3 for derivation.

    INPUTS:
    - hidden: node values in hidden layer (shape: (H,))
    - outputs: node values in output layer (shape: (K,))
    - target: feature vector for training example (shape: (K,))
    - w_out: weights connecting hidden and output layer (shape: (K,H))

    OUTPUTS:
    - partial_outs (ndarray): partial errors for output layer (shape: (K,))
    - partial_hiddens (ndarray): partial errors for hidden layer (shape: (H,))
    """

    # backprop partial errors for output layer
    partial_outs = outputs * (1 - outputs) * (target - outputs)

    # backprop partial errors for hidden layer
    partial_hiddens = hidden * (1 - hidden) * (partial_outs @ w_out)

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
