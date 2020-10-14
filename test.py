import numpy as np
from train import train, forward
from utils import parse_filename, pretty_print
from data import build_data_from_arff
from random import shuffle


def k_fold_cross_val(data, k=10, shuffle_data=True, n_hidden=None,
                     learn_rate=0.1, n_epochs=500, verbose=True):

    """
    Performs k-fold cross-validation. Returns average accuracy over all folds.

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

    # Group examples s.t. fold sizes differ by at most 1.
    # Assign [N = floor(len(data) / k)] examples to each fold.
    # Assign one additional example to [r = len(data) % k] folds.
    groups = []
    size_per_group = len(data) // k
    r = len(data) % k
    start = 0
    for i in range(k):
        n_examples = size_per_group + (i < r)
        groups.append(data[start: start + n_examples])
        start += n_examples

    # cross validation each fold
    total_correct = total_seen = 0
    for i in range(k):
        print("\nUsing group {} of {} as test data".format(i+1, k))
        train_data = [x for group in groups[:i] + groups[i+1:] for x in group]
        test_data = groups[i]
        # pretty_print(train_data)
        # pretty_print(test_data)

        n_correct, n_seen, _, _ = test(train_data, test_data, n_hidden=n_hidden)
        total_correct += n_correct
        total_seen += n_seen

    avg_acc = 100 * total_correct / total_seen
    print("Average accuracy: {:.2f}% ({}:{})"
          .format(avg_acc, total_correct, total_seen - total_correct))

    return avg_acc


def test(train_data, test_data=None, n_hidden=None, learn_rate=0.1,
         n_epochs=500, verbose=True):
    """
    Uses <test_data> to evaluate classification accuracy of network trained on
    <train_data>.

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

    # classify test data using trained weights/biases
    weights_biases = train(train_data, n_hidden, learn_rate, n_epochs)
    if n_hidden == 0:
        w_out, b_out = weights_biases
        w_h = b_h = None
    else:
        w_out, b_out, w_h, b_h = weights_biases

    n_correct = 0
    n_seen = len(test_data)
    for x, target in test_data:
        _, sigma_out = forward(x, w_out, b_out, w_h, b_h)
        pred = np.argmax(sigma_out)
        n_correct += target[pred][0]

    if verbose:
        print("Percent classified correctly: {:.2f}% ({}:{})"
              .format(100 * n_correct / n_seen, n_correct, n_seen - n_correct))

    return n_correct, n_seen, w_h, w_out


if __name__ == "__main__":
    filename = parse_filename()
    data = build_data_from_arff(filename)
    # test(data, n_epochs=300)
    avg_acc = k_fold_cross_val(data, k=10, n_hidden=0)
