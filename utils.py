import argparse


def parse_filename():
    """
    Parses path/to/data.arff from command line.
    """
    parser = argparse.ArgumentParser(description="Naive Bayes classifier",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("data", type=str, help="path to .arff file")
    args = vars(parser.parse_args())
    return args["data"]
