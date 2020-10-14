# TODO handle zero hidden layers
# TODO update comments for recent changes

import numpy as np
from utils import parse_filename
from data import build_data_from_arff


def train(train_data, n_hidden="a", learn_rate=0.1, n_epochs=500):
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

    if n_hidden == "a":
        # set hidden layer size to aveage of input/output layer size
        n_hidden = (n_in + n_out) // 2

    # init hidden layer
    w_h = None
    b_h = None
    if (n_hidden > 0):
        w_h = _init_weights(n_hidden, n_in)     # H x N
        b_h = _init_weights(n_hidden, 1)        # H x 1
    else:
        n_hidden = n_in

    # init output layer
    w_out = _init_weights(n_out, n_hidden)  # K x H
    b_out = _init_weights(n_out, 1)         # K x 1

    for epoch in range(n_epochs):
        for x, target in train_data:
            gradients = _backprop(x, target, w_out, b_out, w_h, b_h)

            # update output layer weights/biases
            w_out += learn_rate * gradients[0]
            b_out += learn_rate * gradients[1]

            # update hidden layer weights/bases
            if (w_h is not None):
                w_h += learn_rate * gradients[2]
                b_h += learn_rate * gradients[3]

    return (w_out, b_out, w_h, b_h) if (w_h is not None) else (w_out, b_out)


def forward(x, w_out, b_out, w_h=None, b_h=None):
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

    if (w_h is not None):
        h = w_h @ x + b_h
        sigma_h = _sigmoid(h)           # H x 1
    else:
        sigma_h = x

    out = w_out @ sigma_h + b_out
    sigma_out = _sigmoid(out)       # K x 1
    return sigma_h, sigma_out


""" PRIVATE METHODS """


def _backprop(x, target, w_out, b_out, w_h=None, b_h=None):
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
    - returns gradients of the weights/biases (if the network doesn't contain a
      hidden layer, then only the output layer gradients are returned)
    """

    sigma_h, sigma_out = forward(x, w_out, b_out, w_h, b_h)

    # ∂L/∂b_out = ∂L/∂sigma_out * ∂sigma_out/∂out * ∂out/∂w_out
    # Notice ∂out/∂w_out is just the one vector since coefficient of b_h
    # in [out = w_h @ x + b_h] is 1.
    # Therefore, ∂L/∂b_out = ∂L/∂sigma_out * ∂sigma_out/∂out = ∂L/∂out
    db_out = (target - sigma_out) * sigma_out * (1 - sigma_out)  # K x 1

    # ∂L/∂w_out = ∂L/∂out * ∂out/∂w_out
    dw_out = db_out @ sigma_h.reshape(1, -1)    # K x H

    if (w_h is None):
        return dw_out, db_out

    # ∂L/∂sigma_h = ∂L/∂out * ∂out/∂sigma_h
    # Note: Mitchell refers to this partial as ∂net_k.
    dsigma_h = np.transpose(w_out) @ db_out     # H x 1

    # ∂L/∂h = ∂L/∂sigma_h * ∂sigma_h/∂h
    # Similar to ∂L/∂b_out, ∂h/∂b_h is the one vector, so ∂L/∂b_h = ∂L/∂h.
    db_h = dsigma_h * sigma_h * (1 - sigma_h)   # H x 1

    # ∂L/∂w_h = ∂L/∂h * ∂h/∂w_h
    dw_h = db_h @ x.reshape(1, -1)               # H x N

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
