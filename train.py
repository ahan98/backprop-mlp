import numpy as np


def train(train_data, n_hidden="a", learn_rate=0.1, n_epochs=500):
    """
    Trains network weights/biases on <train_data> using stochastic gradient
    descent and backpropagation.

    Returns the learned weights/biases.

    INPUTS:
    - train_data (list): each example is a tuple (x, t) denoting input/target
      feature vectors, respectively
    - n_hidden (str): num. of nodes in hidden layer
        NOTE: just like Weka, we assume the default wildcard value "a", i.e.,
        average of input and output layer size
    - learn_rate (float): step size for gradient update
    - n_epochs (int): num. of iterations using all of <train_data>

    OUTPUTS:
    - (w_out, b_out, w_h, b_h): updated weights/biases for output/hidden layers
        NOTE: w_h and b_h are null if there is no hidden layer
    """

    # layer sizes
    n_in, n_out = len(train_data[0][0]), len(train_data[0][1])
    if n_hidden == "a":
        n_hidden = (n_in + n_out) // 2  # average of input/output size
    elif n_hidden.isdigit():
        n_hidden = int(n_hidden)
    else:
        raise ValueError("Invalid hidden layer size {}".format(n_hidden))

    # init hidden layer
    w_h = None
    b_h = None
    if (n_hidden > 0):
        w_h = _init_weights(n_hidden, n_in)     # H x N
        b_h = _init_weights(n_hidden, 1)        # H x 1
    else:
        # if there is no hidden layer, input layer replaces hidden layer,
        # since input layer now directly precedes output layer
        n_hidden = n_in

    # init output layer (note: if there is no hidden layer, K == N)
    w_out = _init_weights(n_out, n_hidden)  # K x H
    b_out = _init_weights(n_out, 1)         # K x 1

    for epoch in range(n_epochs):
        for x, target in train_data:
            dw_out, db_out, dw_h, db_h = _backprop(x, target, w_out, b_out, w_h, b_h)

            # update output layer weights/biases
            w_out += learn_rate * dw_out
            b_out += learn_rate * db_out

            # update hidden layer weights/bases
            if (w_h is not None):
                w_h += learn_rate * dw_h
                b_h += learn_rate * db_h

    return (w_out, b_out, w_h, b_h)


def forward(x, w_out, b_out, w_h=None, b_h=None):
    """
    Performs one forward pass through the network. Returns sigmoid-activated
    outputs for hidden layer (or non-activated input layer, if there is no
    hidden layer) and output layer.

    INPUTS:
    - x (ndarray): input feature vector (shape: H x 1)
    - w_out (ndarray): output layer weights (shape: K x H)
    - b_out (ndarray): output layer biases (shape: K x 1)
    - w_h (ndarray): hidden layer weights (shape: H x N)
    - b_h (ndarray): hidden layer biases (shape: H x 1)

    OUTPUTS:
    - sigma_h, sigma_out: respective outputs for hidden/output layers
        NOTE: sigma_h is the original input layer if there is no hidden layer
    """

    sigma_h = _sigmoid(w_h @ x + b_h) if (w_h is not None) else x   # H x 1
    sigma_out = _sigmoid(w_out @ sigma_h + b_out)                   # K x 1
    return sigma_h, sigma_out


""" PRIVATE METHODS """


def _backprop(x, target, w_out, b_out, w_h=None, b_h=None):
    """
    Performs one feed-forward pass, then backpropagates the output to compute
    the partials of each layer's weights/biases.

    INPUTS:
    - x (ndarray): input feature vector (shape: H x 1)
    - target (ndarray): ground truth labels for x
    - w_out (ndarray): output layer weights (shape: K x H)
    - b_out (ndarray): output layer biases (shape: K x 1)
    - w_h (ndarray): hidden layer weights (shape: H x N)
    - b_h (ndarray): hidden layer biases (shape: H x 1)

    OUTPUT:
    - (dw_out, db_out, dw_h, db_h): weight/bias gradients for output/hidden
        NOTE: dw_h and db_h are null if there is no hidden layer
    """

    # If there is no hidden layer, <sigma_h> is just the input layer (see
    # <forward>). In this case, we return immediately after computing gradients
    # for output layer, since there are no more layers to consider.
    sigma_h, sigma_out = forward(x, w_out, b_out, w_h, b_h)

    # Backprop sigma_out = sigmoid(out) = sigmoid(w_out @ sigma_h + b_out).
    # We denote the loss function by L.

    # ∂L/∂b_out = ∂L/∂sigma_out * ∂sigma_out/∂out * ∂out/∂b_out
    # Notice ∂out/∂b_out is the one vector since coefficient of b_out is 1.
    # Therefore, ∂L/∂b_out = ∂L/∂sigma_out * ∂sigma_out/∂out = ∂L/∂out.
    db_out = (target - sigma_out) * sigma_out * (1 - sigma_out)  # K x 1

    # ∂L/∂w_out = ∂L/∂out * ∂out/∂w_out
    dw_out = db_out @ sigma_h.reshape(1, -1)    # K x H

    if (w_h is None):
        return (dw_out, db_out, None, None)

    # Backprop sigma_h = sigmoid(h) = sigmoid(w_h @ x + b_h).
    # This is only necessary if there is a hidden layer.

    # ∂L/∂sigma_h = ∂L/∂out * ∂out/∂sigma_h
    # Note: Mitchell refers to this partial as ∂net_k.
    dsigma_h = np.transpose(w_out) @ db_out     # H x 1

    # ∂L/∂h = ∂L/∂sigma_h * ∂sigma_h/∂h
    # Similar to ∂L/∂b_out, ∂h/∂b_h is the one vector, so ∂L/∂b_h = ∂L/∂h.
    db_h = dsigma_h * sigma_h * (1 - sigma_h)   # H x 1

    # ∂L/∂w_h = ∂L/∂h * ∂h/∂w_h
    dw_h = db_h @ x.reshape(1, -1)              # H x N

    return (dw_out, db_out, dw_h, db_h)


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
