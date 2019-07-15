import copy
from math import log
import numpy as np
import pandas as pd

from datagen import create_data

np.random.seed(1234)


# ID3 decision_tree constructing
class Node:
    def __init__(self, direction=None, leaf=True, label=None, feature_name=None, feature_index=None, condition=None):
        self.direction = direction
        self.leaf = leaf
        self.label = label
        self.feature_name = feature_name
        self.feature_index = feature_index
        self.condition_value = condition
        self.condition = None if (condition is None) else ">{:.3f}".format(condition)
        self.tree = {}
        self.result = {
            'leaf or not': self.leaf,
            'label': self.label,
            'feature_name': self.feature_name,
            'condition': self.condition,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features, index):
        if self.leaf is True:
            return self.label
        if features[index.get(self.feature_name)] >= self.condition_value:
            return self.tree['Yes'].predict(features, index)
        else:
            return self.tree['No'].predict(features, index)


class my_DecisionTreeClassifier:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    # calculate information entropy
    @staticmethod
    def calc_entropy(y_train):
        data_length = len(y_train)
        label_count = {}
        # calculate numbers of all labels
        for i in range(data_length):
            label = y_train[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2) for p in label_count.values()])
        return ent

    # calculate condition entropy
    def cond_entropy(self, x_train, y_train, axis=0, feature_list=None, threshold_dict=None):
        data_length = len(x_train)
        feature_sets = {}
        # continuous value processing
        if isinstance(x_train[0][axis], str) is False:
            # generate n-1 thresholds
            threshold = []
            gain_trans = []
            ent_data = self.calc_entropy(y_train)
            x_train_ = list(sorted(x_train[:, axis], reverse=False))
            for i in range(data_length - 1):
                threshold.append((x_train_[i] + x_train_[i + 1]) / 2)

            for j in range(len(threshold)):
                index = np.where(x_train[:, axis] < threshold[j])
                below = x_train[index[0], :]
                below_y = y_train[index[0], :]
                below_ent = self.calc_entropy(below_y)
                index = np.where(x_train[:, axis] >= threshold[j])
                upper = x_train[index[0], :]
                upper_y = y_train[index[0], :]
                upper_ent = self.calc_entropy(upper_y)
                gain = ent_data - (len(below) * below_ent + len(upper) * upper_ent) / data_length
                gain_trans.append(gain)

            best_thresh_index = int(np.argmax(gain_trans))
            best_thresh = threshold[best_thresh_index]

            threshold_dict[feature_list[axis]] = best_thresh
            cond_ent = np.max(gain_trans)

        else:
            for i in range(data_length):
                feature = x_train[i][axis]
                if feature not in feature_sets:
                    feature_sets[feature] = []
                feature_sets[feature].append(x_train[i])
            cond_ent = sum(
                [len(p) / data_length * self.calc_entropy(p) for p in feature_sets.values()]
            )  # p is a sub_datasets
        return cond_ent

    def info_gain_train(self, x_train, y_train, features, threshold_dict=None):
        count = len(features)

        ent = self.calc_entropy(y_train)
        best_feature = []
        for c in range(count):
            cond_ent = self.cond_entropy(x_train, y_train, axis=c,
                                         feature_list=features,
                                         threshold_dict=threshold_dict
                                         )
            c_info_gain = self.info_gain(ent, cond_ent)
            best_feature.append((c, c_info_gain))
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, x_train, y_train, features):

        threshold_dict = {}
        data_length = len(y_train)
        y_train_rs = y_train.reshape((data_length, 1))
        # if datasets are from same label, then mark T as single_node tree and choose its label as tree label
        if len(np.unique(y_train)) == 1:
            return Node(leaf=True, label=y_train_rs[0][0])

        # if datasets is empty then take the label with maximum number from its parent node as label
        if len(features) == 0:
            return Node(leaf=True,
                        label=np.argmax(np.bincount(y_train_rs.flatten())))
        # calculate the max info_gain
        max_feature_index, max_info_gain = self.info_gain_train(x_train, y_train_rs, features, threshold_dict)
        max_feature_name = features[max_feature_index]

        # if info_gain less than threshold eta then set it as single_node tree,
        # take the label with maximum number from its parent node as label
        if max_info_gain < self.epsilon:
            return Node(leaf=True,
                        label=np.argmax(np.bincount(y_train_rs.flatten())))

        # create sub_node
        node_tree = Node(leaf=False,
                         feature_name=max_feature_name, feature_index=max_feature_index,
                         condition=threshold_dict[max_feature_name])

        # feature_list = train_data[max_feature_name].value_counts().index
        # for f in feature_list:
        #     # classify all sub_class and delete the data of its node's label
        #     sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)
        #
        #     # generate tree recursivly  递归
        #     sub_tree = self.train(sub_train_df)
        #     node_tree.add_node(f, sub_tree)

        # sub_train_df = train_data.loc[train_data[max_feature_name] < threshold_dict[max_feature_name]].drop(
        #     [max_feature_name], axis=1)
        index_no = np.where(x_train[:, max_feature_index] < threshold_dict[max_feature_name])
        sub_x_train_no = np.delete((x_train[index_no[0], :]), max_feature_index, axis=1)
        # sub_x_train_no = x_train[index_no[0], :]
        sub_y_train_no = y_train_rs[index_no[0], :]
        index_yes = np.where(x_train[:, max_feature_index] >= threshold_dict[max_feature_name])
        sub_x_train_yes = np.delete((x_train[index_yes[0], :]), max_feature_index, axis=1)
        # sub_x_train_yes = x_train[index_yes[0], :]
        sub_y_train_yes = y_train_rs[index_yes[0], :]
        features.remove(features[max_feature_index])
        # deep copy for two
        features_1 = copy.deepcopy(features)
        features_2 = copy.deepcopy(features)
        # print(sub_x_train_no.shape, sub_x_train_yes.shape, len(features_1), max_feature_index)
        if len(sub_x_train_no) > 0:
            sub_tree_no = self.train(sub_x_train_no, sub_y_train_no, features_1)
            node_tree.add_node('No', sub_tree_no)
        if len(sub_x_train_yes) > 0:
            sub_tree_yes = self.train(sub_x_train_yes, sub_y_train_yes, features_2)
            node_tree.add_node('Yes', sub_tree_yes)

        return node_tree

    def fit(self, x_train, y_train, features=None):
        self.x_train = x_train
        self.y_train = y_train
        self.features = features
        self.index = {}
        for i in range(len(features)):
            self.index[features[i]] = i
        self._tree = self.train(self.x_train, self.y_train, self.features)

        return self._tree

    def predict(self, x_test, index):
        data_length = len(x_test)
        result = []
        for i in range(data_length):
            result.append(self._tree.predict(x_test[i], index))
        return result

    def score(self, x_test, y_test, index):
        result = self.predict(x_test, index)
        result = np.array(result).reshape(-1, )
        y_test_rs = np.reshape(y_test, (-1,))
        count = 0
        for i in range(len(result)):
            if result[i] == y_test_rs[i]:
                count += 1
        return count / len(result), result


if __name__ == "__main__":
    col = ['class',
           'Alcohol',
           'Malic acid',
           'Ash',
           'Alcalinity of ash',
           'Magnesium',
           'Total phenols',
           ' Flavanoids',
           'Nonflavanoid phenols',
           'Proanthocyanins',
           'Color intensity',
           'Hue',
           'OD280/OD315 of diluted wines',
           'Proline']

    data = pd.read_csv('wine.data', names=col)
    DTree = my_DecisionTreeClassifier()
    train_data = data.sample(frac=0.7)

    test_data = data[~data.index.isin(train_data.index)]
    x_train = np.array(train_data.iloc[:, 1:])
    y_train = np.array(train_data.iloc[:, 0]).reshape(len(x_train), 1)
    x_test = np.array(test_data.iloc[:, 1:])
    y_test = np.array(test_data.iloc[:, 0]).reshape(len(x_test), 1)
    print(DTree.fit(x_train, y_train, list(train_data.columns[1:])))

    score, result = DTree.score(x_test, y_test, DTree.index)
    print(result)
    print(score)
