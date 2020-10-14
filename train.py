# TODO handle zero hidden layers

import numpy as np
from utils import parse_filename
from data import build_data_from_arff


def train(train_data, n_hidden=None, learn_rate=0.1, n_epochs=500):
    """
    Trains network weights/biases on <train_data> using stochastic gradient
    descent and backpropagation.

    INPUTS:
    - train_data (list): each example is a tuple (x, t) denoting input/target
      feature vectors, respectively
    - n_hidden (int): num. of nodes in hidden layer
    - learn_rate (int): step size for gradient update
    - n_epochs (int): num. of times each example in <train_data> is used once
    """

    n_in, n_out = len(train_data[0][0]), len(train_data[0][1])

    if n_hidden is None:
        # set hidden layer size to aveage of input/output layer size
        n_hidden = (n_in + n_out) // 2

    # init random weight matrices
    w_h = _init_weights(n_hidden, n_in)     # H x N
    w_out = _init_weights(n_out, n_hidden)  # K x H

    # init random bias vectors
    b_h = _init_weights(n_hidden, 1)  # H x 1
    b_out = _init_weights(n_out, 1)   # K x 1

    for epoch in range(n_epochs):
        for x, target in train_data:
            dw_out, db_out, dw_h, db_h = _backprop(x, w_h, b_h, w_out, b_out, target)

            # update output layer weights/biases
            w_out += learn_rate * dw_out
            b_out += learn_rate * db_out

            # update hidden layer weights/biases
            w_h += learn_rate * dw_h
            b_h += learn_rate * db_h

    return w_h, b_h, w_out, b_out


def forward(x, w_h, b_h, w_out, b_out):
    """
    Performs one forward pass through the network. Returns sigmoid activated
    outputs for hidden and output layer.

    INPUTS:
    - x (ndarray): input feature vector (shape: H x 1)
    - w_h (ndarray): hidden layer weights (shape: H x N)
    - b_h (ndarray): hidden layer biases (shape: H x 1)
    - w_out (ndarray): output layer weights (shape: K x H)
    - b_out (ndarray): output layer biases (shape: K x 1)

    OUTPUTS:
    - sigma_h, sigma_out: respective outputs for hidden/output layers
    """

    h = w_h @ x + b_h
    sigma_h = _sigmoid(h)           # H x 1
    out = w_out @ sigma_h + b_out
    sigma_out = _sigmoid(out)       # K x 1
    return sigma_h, sigma_out


""" PRIVATE METHODS """


def _backprop(x, w_h, b_h, w_out, b_out, target):
    """
    Performs one forward pass through the network. Uses network outputs to
    compute gradients for each layers' weights/biases.

    INPUTS:
    - x (ndarray): input feature vector (shape: H x 1)
    - w_h (ndarray): hidden layer weights (shape: H x N)
    - b_h (ndarray): hidden layer biases (shape: H x 1)
    - w_out (ndarray): output layer weights (shape: K x H)
    - b_out (ndarray): output layer biases (shape: K x 1)
    - target (ndarray): ground truth labels for x

    OUTPUT:
    - dw_out, db_out, dw_h, db_h: gradients of the weights/biases
    """

    sigma_h, sigma_out = forward(x, w_h, b_h, w_out, b_out)

    # ∂L/∂b_out = ∂L/∂sigma_out * ∂sigma_out/∂out * ∂out/∂w_out
    # Notice ∂out/∂w_out is just the one vector since coefficient of b_h
    # in w_h @ x + b_h is 1.
    # Therefore, ∂L/∂b_out = ∂L/∂sigma_out * ∂sigma_out/∂out = ∂L/∂out
    db_out = (target - sigma_out) * sigma_out * (1 - sigma_out)  # K x 1

    # ∂L/∂w_out = ∂L/∂out * ∂out/∂w_out
    dw_out = db_out @ sigma_h.reshape(1, -1)    # K x H

    # ∂L/∂sigma_h = ∂L/∂out * ∂out/∂sigma_h
    # Note: Mitchell refers to this partial as ∂net_k.
    dsigma_h = np.transpose(w_out) @ db_out     # H x 1

    # ∂L/∂h = ∂L/∂sigma_h * ∂sigma_h/∂h
    # Similar to ∂L/∂b_out, ∂h/∂b_h is the one vector, so ∂L/∂b_h = ∂L/∂h.
    db_h = dsigma_h * sigma_h * (1 - sigma_h)   # H x 1

    # ∂L/∂w_h = ∂L/∂h * ∂h/∂w_h
    dw_h = db_h @ np.transpose(x)               # H x N

    return dw_out, db_out, dw_h, db_h


def _init_weights(n_rows, n_cols, std_dev=0.5):
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
