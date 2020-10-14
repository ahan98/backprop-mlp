import re
import numpy as np
from utils import parse_arff_file


def build_data_from_arff(arff_file):
    """
    Parses data examples in .arff file. Feature vectors are implemented as
    column vectors. Entries are boolean, except when attributes are missing
    (see below). Indexes map sequentially to the attribute values, i.e., in the
    order they appear in <arff_file>.

    INPUTS:
    - arff_file (str): path/to/data.arff

    OUPUTS:
    - data [list]: each example is stored as a tuple of feature vectors (x, t),
      denoting input/target feature vectors respectively (note: t is one-hot)
    """

    lines = None
    with open(arff_file, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]

    n_attr_vals = 0   # total num. attribute values
    n_class_vals = 0  # num. values in last seen attribute (i.e., the class)
    data = []

    # maps value of i-th attribute to sequential index (0-indexed)
    # e.g., in <weather.nominal.arff>, attr_val_to_index[1]["hot"] = 3
    attr_val_to_index = []

    for line in lines:
        if line[0] == "%":  # ignore .arff comments
            continue

        stripped = re.sub(r"[\n\t\s]*", "", line)  # remove whitespace
        stripped = stripped.lower()

        # parse attribute and class names/values
        if stripped[:10] == "@attribute":
            attr_vals_start = stripped.index("{")
            attr_vals = stripped[attr_vals_start + 1: -1].split(",")

            # map attribute values to sequential index
            attr_val_dict = {attr_val: (idx + n_attr_vals)
                             for idx, attr_val in enumerate(attr_vals)}
            attr_val_to_index.append(attr_val_dict)

            n_attr_vals += len(attr_vals)
            n_class_vals = len(attr_vals)

        # convert each example into boolean feature vectors
        elif stripped[0] != "!" and stripped[0] != "@":
            features = stripped.split(",")
            attrs = [0 for _ in range(n_attr_vals)]
            for i, val in enumerate(features):
                if val == "?":
                    # distribute across all values when attribute is missing
                    total_features = len(attr_val_to_index[i])
                    for idx in attr_val_to_index[i].values():
                        attrs[idx] = 1 / total_features
                else:
                    # present attribute values are set to 1, and 0 otherwise
                    idx = attr_val_to_index[i][val]
                    attrs[idx] = 1

            # split into input and target vectors
            # reshaped as columns for matrix algebra (see <train.py>)
            x = np.array(attrs[:-n_class_vals]).reshape(-1, 1)
            target = np.array(attrs[-n_class_vals:]).reshape(-1, 1)
            data.append((x, target))

    return data


if __name__ == "__main__":
    arff_file = parse_arff_file()
    data = build_data_from_arff(arff_file)
    for d in data:
        print(d)
