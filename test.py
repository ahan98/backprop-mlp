# TODO cross validation and hold-one-out

import numpy as np
from train import train, forward
from utils import parse_filename
from data import build_data_from_arff
from random import shuffle


def k_fold_cross_val(data, k=10, shuffle_data=True):
    """
    INPUTS:
    - data (list): list of examples, where each example is a tuple of input and
      target feature vectors
    - k (int): number of cross-validation folds
    - shuffle_data (bool): shuffles data in place if True

    OUTPUTS:
    - avg_acc (float): average test accuracy from all k folds
    """

    if shuffle_data:
        shuffle(data)

    # create folds s.t. fold sizes differ by at most 1
    # first, each fold has N = floor(len(data) / k) elements
    # the first r = len(data) % k folds have N + 1 elements
    groups = []
    size_per_group = len(data) // k
    r = len(data) % k
    for i in range(k):
        groups.append(data[i: (i + size_per_group + (i < r))])

    # cross validation each fold
    cumulative_correct = cumulative_total = 0
    for i in range(k):
        print("\nUsing group {} of {} as test data".format(i+1, k))
        train_data = [data for group in groups[:i] + groups[i+1:] for data in group]
        test_data = groups[i]
        n_correct, n_total, _, _ = test(train_data, test_data)

        cumulative_correct += n_correct
        cumulative_total += n_total

    avg_acc = 100 * cumulative_correct / cumulative_total
    print("Average accuracy: {:.2f}% ({}:{})"
          .format(avg_acc, cumulative_correct, cumulative_total - cumulative_correct))

    return avg_acc


def test(train_data, test_data=None, n_hidden=None, learn_rate=0.1,
         n_epochs=500, verbose=True):
    """
    INPUTS:
    - train_data (list): list of train data, stored as tuples of input/target
      feature vectors
    - test_data (list): list of test data, in same format as train data
    - n_hidden (int): size of hidden layer
    - learn_rate (float): learning rate for grad. descent
    - n_epochs (int): total iterations over all train data
    - verbose (bool): prints test accuracy if True

    OUTPUTS:
    - returns test accuracy and trained weights
    """

    # test data re-uses train data, if unspecified
    if test_data is None:
        test_data = train_data

    # classify test data using trained weights
    w_hidden, w_out = train(train_data, n_hidden, learn_rate, n_epochs)
    n_correct = 0
    n_total = len(test_data)
    for input, target in test_data:
        _, out = forward(input, w_hidden, w_out)
        predicted_class_val = np.argmax(out)
        n_correct += target[predicted_class_val]

    if verbose:
        print("Percent classified correctly: {:.2f}% ({}:{})"
              .format(100 * n_correct / n_total, n_correct, n_total - n_correct))

    return n_correct, n_total, w_hidden, w_out


if __name__ == "__main__":
    filename = parse_filename()
    data = build_data_from_arff(filename)
    # test(data)
    avg_acc = k_fold_cross_val(data, k=10)
