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
    w_h = _init_weights(n_hidden, n_in)  # H x N
    w_out = _init_weights(n_out, n_hidden)    # K x H

    # initialize bias vectors
    b_h = _init_weights(n_hidden, 1)  # H x 1
    b_out = _init_weights(n_out, 1)   # K x 1

    for epoch in range(n_epochs):
        for x, target in train_data:
            # TODO think about how to make shape change more efficient
            # once everything else works, try with shape (N,) instead of (1,N)
            # i dont think moving it into data.py makes sense
            x = np.array(x).reshape(-1, 1)    # N x 1
            target = np.array(target).reshape(-1, 1)  # K x 1

            dwout, dbout, dwh, dbh = _backprop(x, w_h, b_h, w_out, b_out, target)
            w_out += learn_rate * dwout
            b_out += learn_rate * dbout

            w_h += learn_rate * dwh
            b_h += learn_rate * dbh

    return w_h, b_h, w_out, b_out


def forward(x, w_h, b_h, w_out, b_out):
    h = _sigmoid(w_h @ x + b_h)      # (H x N) x (N x 1) = H x 1
    out = _sigmoid(w_out @ h + b_out)  # (K x H) x (H x 1) = K x 1
    return h, out


""" PRIVATE METHODS """


def _backprop(x, w_h, b_h, w_out, b_out, target):
    h, out = forward(x, w_h, b_h, w_out, b_out)
    dloss_dsigma = (target - out)    # K x 1
    dsigma_dout = (out) * (1 - out)  # K x 1

    dout_dwout = np.ones(len(out)).reshape(-1, 1) @ h.reshape(1, -1)  # K x H
    dbout = dloss_dsigma * dsigma_dout  # K x 1
    dwout = dbout * dout_dwout  # K x H

    dout_dsigmah = w_out  # K x H
    dloss_dsigmah = dloss_dsigma * dsigma_dout * dout_dsigmah  # K x 1
    dloss_dsigmah = dloss_dsigmah.sum(axis=0).reshape(-1, 1)  # H x 1

    dsigmah_dh = h * (1 - h)  # H x 1
    dh_dwh = x
    dbh = dloss_dsigmah * dsigmah_dh  # H x 1
    dwh = dbh @ dh_dwh.reshape(1, -1)

    return dwout, dbout, dwh, dbh


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
