# TODO comments

import numpy as np
from utils import parse_filename
from data import build_data_from_arff


def train(train_data, n_hidden=None, learn_rate=0.1, n_epochs=500):
    n_in, n_out = len(train_data[0][0]), len(train_data[0][1])

    if n_hidden is None:
        # set hidden layer size to aveage of input/output layer size
        n_hidden = (n_in + n_out) // 2

    # initialize weight matrices
    w_h = _init_weights(n_hidden, n_in)  # H x N
    w_out = _init_weights(n_out, n_hidden)    # K x H

    # initialize bias vectors
    b_h = _init_weights(n_hidden, 1)  # H x 1
    b_out = _init_weights(n_out, 1)   # K x 1

    for epoch in range(n_epochs):
        for x, target in train_data:
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

    dsigma_out = (target - out)    # K x 1
    dsigma_out_dout = (out) * (1 - out)  # K x 1
    db_out = dsigma_out * dsigma_out_dout  # K x 1

    dout_dw_out = np.ones(len(out)).reshape(-1, 1) @ h.reshape(1, -1)  # K x H
    dw_out = db_out * dout_dw_out  # K x H

    dout_dsigma_h = w_out  # K x H
    dsigma_h = dsigma_out * dsigma_out_dout * dout_dsigma_h  # K x 1
    dsigma_h = dsigma_h.sum(axis=0).reshape(-1, 1)  # H x 1
    dsigmah_dh = h * (1 - h)  # H x 1
    db_h = dsigma_h * dsigmah_dh  # H x 1

    dh_dw_h = x
    dw_h = db_h @ dh_dw_h.reshape(1, -1)  # H x N

    return dw_out, db_out, dw_h, db_h


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
