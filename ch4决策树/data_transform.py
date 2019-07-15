from math import log

import numpy as np
import pandas as pd

np.random.seed(1234)


def info_gain(ent, cond_ent):
    return ent - cond_ent


def calc_entropy(datasets):
    data_length = len(datasets)
    label_count = {}
    # calculate numbers of all labels
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
    return ent


def cond_entropy(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    # continuous value processing
    if isinstance(datasets[0][axis], str) is False:
        # generate n-1 thresholds
        threshold = []
        for i in range(data_length - 2):
            threshold.append((datasets[i][axis] + datasets[i + 1][axis]) / 2)
        gain_trans = []
        ent_data = calc_entropy(datasets)
        for j in range(len(threshold)):
            below = datasets[:j + 1]
            upper = datasets[j + 1:]
            ent_below = calc_entropy(below)
            ent_upper = calc_entropy(upper)
            gain = ent_data - ((j + 1) * ent_below + (data_length - j) * ent_upper) / data_length
            gain_trans.append(gain)
        best_thresh = threshold[int(np.argmax(gain_trans))]

        # classify into two kinds
        for i in range(data_length):
            if datasets[i][axis] < best_thresh:
                datasets[i][axis] = 0
            else:
                datasets[i][axis] = 1
        cond_ent = np.max(gain_trans)
    else:
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum(
            [len(p) / data_length * calc_entropy(p) for p in feature_sets.values()]
        )  # p is a sub_datasets
    return cond_ent


def info_gain_train(datasets):
    count = len(datasets[0]) - 1
    ent = calc_entropy(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_entropy(datasets, axis=c))
        best_feature.append((c, c_info_gain))
    best_ = max(best_feature, key=lambda x: x[-1])
    return best_


data = [0.244, 0.29, 0.351, 0.381, 0.420, 0.459, 0.518, 0.574, 0.600, 0.621, 0.636, 0.648, 0.661, 0.681, 0.685, 0.746]

dataset = []
data_len = len(data)
for i in range(data_len):
    if np.random.random(1) > 0.5:
        dataset.append([data[i], 1])
    else:
        dataset.append([data[i], 0])

print(info_gain_train(np.array(dataset)))
