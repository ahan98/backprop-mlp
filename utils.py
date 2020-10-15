import argparse


def parse_filename():
    """
    Parses path/to/data.arff from command line.
    """
    parser = argparse.ArgumentParser(description="Trains simple MLP on nominal data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # required args
    parser.add_argument("data", type=str, help="path to .arff data set")

    # optional args

    parser.add_argument("-H", "--hidden", metavar="\b", type=str, default="a",
                        help="size of hidden layer")

    parser.add_argument("-K", "--k_folds", metavar="\b", type=int, default=None,
                        help="k-fold cross validation")

    parser.add_argument("-E", "--epochs", metavar="\b", type=int, default=500,
                        help="number of epochs")

    parser.add_argument("-L", "--learn_rate", metavar="\b", type=float, default=0.1,
                        help="learning rate")

    args = vars(parser.parse_args())
    return args


def pretty_print(data):
    """
    Print data examples in readable format.
    """
    for x, target in data:
        print(x.reshape(-1), target.reshape(-1))


def print_model_params(n_hidden, n_epochs, learn_rate, k_folds):
    print("\nHidden layer size:", n_hidden)
    print("Epochs:", n_epochs)
    print("Learning rate:", learn_rate)
    if k_folds is not None:
        print("Cross validation folds:", k_folds)
