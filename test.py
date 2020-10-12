# TODO cross validation and hold-one-out

import numpy as np
from train import backprop, forward
from utils import parse_filename, accuracy
from data import build_data_from_arff
from random import shuffle


def k_fold_cross_val(data, k=None, shuffle_data=True):
    if shuffle_data:
        shuffle(data)

    if k is None:
        k = len(data)

    groups = []
    size_per_group = len(data) // k
    r = len(data) % k
    for i in range(k):
        groups.append(data[i: (i + size_per_group + (i < r))])

    best_acc = (0, 0, 0)  # (% correct, n_correct, n_total)
    best_weights = None
    for i in range(k):
        train_data = groups[i]
        test_data = [y for x in groups[:i] + groups[i+1:] for y in x]
        n_correct, n_total, w_hidden, w_out = test(train_data, test_data)
        acc = n_correct / n_total
        if acc > best_acc[0]:
            best_acc = (acc, n_correct, n_total)
            best_weights = (w_hidden, w_out)

    return best_acc, best_weights


def test(train_data, test_data, n_hidden=None, learn_rate=0.1, n_epochs=300):
    w_hidden, w_out = backprop(train_data, n_hidden, learn_rate, n_epochs)
    preds = []
    targets = []
    for input, target in test_data:
        _, out = forward(input, w_hidden, w_out)
        preds.append(np.argmax(out))
        targets.append(target.index(1))

    n_correct, n_total = accuracy(preds, targets)
    return n_correct, n_total, w_hidden, w_out


if __name__ == "__main__":
    filename = parse_filename()
    data = build_data_from_arff(filename)
    best_acc, best_weights = k_fold_cross_val(data, k=10)
    print(best_acc)
