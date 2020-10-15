import numpy as np
from train import train, forward
from utils import parse_filename, pretty_print, print_model_params
from data import build_data_from_arff
from random import shuffle


def k_fold_cross_val(data, k=10, n_hidden=None, learn_rate=0.1, n_epochs=500,
                     verbose=True, shuffle_data=True):

    """
    Performs k-fold cross-validation. Returns average accuracy over all folds.

    INPUTS:
    - data (list): list of examples, where each example is a tuple of input and
      target feature vectors
    - n_hidden (int): size of hidden layer
    - learn_rate (float): learning rate for grad. descent
    - n_epochs (int): total iterations over all train data
    - verbose (bool): prints test accuracy if True
    - shuffle_data (bool): shuffles data in-place if True

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
    total_correct = total_data = 0
    for i in range(k):
        print("\nUsing group {} of {} as test data".format(i+1, k))
        train_data = [x for group in groups[:i] + groups[i+1:] for x in group]
        test_data = groups[i]
        # pretty_print(train_data)
        # pretty_print(test_data)

        n_correct, n_data, _ = test(train_data, test_data, n_hidden=n_hidden)
        total_correct += n_correct
        total_data += n_data

    avg_acc = 100 * total_correct / total_data
    print("\nAverage accuracy: {:.2f}% ({}:{})"
          .format(avg_acc, total_correct, total_data - total_correct))

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
    - returns test accuracy and trained weights/biases
    """

    # test data re-uses train data, if unspecified
    if test_data is None:
        test_data = train_data

    # classify test data using trained weights/biases
    w_out, b_out, w_h, b_h = train(train_data, n_hidden, learn_rate, n_epochs)
    n_correct, n_data = 0, len(test_data)
    for x, target in test_data:
        _, sigma_out = forward(x, w_out, b_out, w_h, b_h)
        pred = np.argmax(sigma_out)
        n_correct += target[pred][0]

    if verbose:
        print("Percent classified correctly: {:.2f}% ({}:{})"
              .format(100 * n_correct / n_data, n_correct, n_data - n_correct))

    return n_correct, n_data, (w_out, b_out, w_h, b_h)


if __name__ == "__main__":
    args = parse_filename()
    data_file = args["data"]
    data = build_data_from_arff(data_file)

    n_hidden = args["hidden"]
    n_epochs = args["epochs"]
    learn_rate = args["learn_rate"]
    k_folds = args["k_folds"]

    print_model_params(n_hidden, n_epochs, learn_rate, k_folds)
    if k_folds is None:
        print("\nTraining with same data set for train/test...")
        _ = test(data, n_hidden=n_hidden, n_epochs=n_epochs,
                 learn_rate=learn_rate)
    else:
        print("\nTraining with {}-fold cross validation...".format(k_folds))
        _ = k_fold_cross_val(data, k=k_folds, n_hidden=n_hidden,
                             n_epochs=n_epochs, learn_rate=learn_rate)
