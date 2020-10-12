# TODO cleanup: comments, variable names

import re
from utils import parse_filename


def build_data_from_arff(filename):
    lines = None
    with open(filename, "r") as f:
        lines = [line for line in f.readlines() if line.strip()]

    n_attr_vals = n_class_vals = 0
    attr_val_to_index = []
    index_to_name = []
    data = []

    for line in lines:
        if line[0] == "%":  # ignore .arff comments
            continue

        stripped = re.sub(r"[\n\t\s]*", "", line)  # remove whitespace
        stripped = stripped.lower()
        # print(stripped)

        # parse attribute and class names/values
        if stripped[:10] == "@attribute":
            attr_vals_start = stripped.index("{")

            attr_name = stripped[10: attr_vals_start]
            index_to_name.append(attr_name)

            attr_vals = stripped[attr_vals_start + 1: -1].split(",")
            attr_val_dict = {attr_val: (idx + n_attr_vals)
                             for idx, attr_val in enumerate(attr_vals)}
            attr_val_to_index.append(attr_val_dict)
            n_attr_vals += len(attr_vals)
            n_class_vals = len(attr_vals)

        # convert each example into boolean vectors of attribute/class values
        elif stripped[0] != "!" and stripped[0] != "@":
            attr_vals = stripped.split(",")
            bools = [0 for _ in range(n_attr_vals)]
            for i, val in enumerate(attr_vals):
                if val == "?":
                    total_attr_vals = len(attr_val_to_index[i])
                    for idx in attr_val_to_index[i].values():
                        bools[idx] = 1 / total_attr_vals
                else:
                    idx = attr_val_to_index[i][val]
                    bools[idx] = 1
            attr_bools, class_bools = bools[:-n_class_vals], bools[-n_class_vals:]
            data.append((attr_bools, class_bools))

    return data


def update_table(table, cumulative, instance, index_to_name, remove):
    class_val = instance[-1]
    sign = -1 if remove else 1  # add or subtract frequencies
    for i, attr_val in enumerate(instance):
        if attr_val == "?":
            continue
        name = index_to_name[i]
        if i == len(instance) - 1:
            table[name][class_val] += sign * 1
            table["TOTAL"] += sign * 1
        else:
            table[name][attr_val][class_val] += sign * 1
            cumulative[name][class_val] += sign * 1


if __name__ == "__main__":
    filename = parse_filename()
    data = build_data_from_arff(filename)
    for d in data:
        print(d)
