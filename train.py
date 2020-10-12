# TODO comments

import numpy as np
from utils import parse_filename
from data import build_data_from_arff


def backprop(train_data, n_hidden=None, learn_rate=0.1, n_epochs=1):
    # initialize weights to small random values
    n_in, n_out = len(train_data[0][0]), len(train_data[0][1])
    if n_hidden is None:
        n_hidden = (n_in + n_out) // 2
    w_hidden = _init_weights(n_hidden, n_in)
    w_out = _init_weights(n_out, n_hidden)
    # print(w_hidden)
    # print(w_out)

    for epoch in range(n_epochs):
        for input, target in train_data:
            hidden, outputs = forward(input, w_hidden, w_out)

            # 1 x K
            partial_outs = np.array([out * (1-out) * (target-out)
                                     for (out, target) in zip(outputs, target)])

            # 1 x H
            partial_hiddens = [h * (1-h) for h in hidden] * (partial_outs @ w_out)

            # update weights
            # print(learn_rate * (partial_outs.reshape(-1, 1) * hidden))
            # print(w_out)
            w_hidden += learn_rate * (partial_hiddens.reshape(-1, 1) * input)
            w_out += learn_rate * (partial_outs.reshape(-1, 1) * hidden)
            # print(w_out)

    return w_hidden, w_out


def forward(input, w_hidden, w_out):
    # inputs is Nx1
    # w_hidden is HxN
    # w_out is KxH

    hidden = _sigmoid(w_hidden @ input)   # H x 1
    outputs = _sigmoid(w_out @ hidden)    # K x 1

    return hidden, outputs


def _init_weights(n_rows, n_cols, std_dev=0.05):
    weights = np.random.randn(n_rows, n_cols) * std_dev
    return weights


def _sigmoid(x):
    return 1/(1 + np.exp(-x))


if __name__ == "__main__":
    filename = parse_filename()
    data = build_data_from_arff(filename)
    for d in data:
        print(d)

    w_hidden, w_out = backprop(data, 3, n_epochs=2)
    print("****")
    print(w_hidden)
    print(w_out)
