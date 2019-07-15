import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from decision_tree import my_DecisionTreeClassifier


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    # data = np.array(df.iloc[:100, [0, 1, -1]])
    data = np.array(df)
    # print(data)
    return data[:, :-1], data[:, -1]


if __name__ == "__main__":
    feature = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # index={}
    # for i in range(len(feature)):
    #     index[feature[i]]=i
    X, y = create_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    y_train = y_train.astype(np.int16).reshape(len(X_train), 1)
    y_test = y_test.astype(np.int16).reshape(len(X_test), 1)
    print(y_train[0])
    print("My Decision Tree model testing: ")
    DTree = my_DecisionTreeClassifier()
    print(DTree.fit(X_train, y_train, feature))
    print(DTree.score(X_test, y_test)[0])
    print("Sklearn Decision Tree model testing: ")
    iris = load_iris()
    DTree = DecisionTreeClassifier(criterion='entropy')
    print(cross_val_score(DTree, iris.data, iris.target, cv=5))
