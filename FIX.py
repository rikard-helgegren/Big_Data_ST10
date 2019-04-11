import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pd.options.mode.chained_assignment = None


def mislabel(labels, p):
    labelSet = list(set(labels))
    for i in range(0, len(labels)):
        if random.uniform(0, 1) < p:
            tmpSet = []
            for j in range(0,len(labelSet)):
                tmpSet.append(labelSet[j])
            tmpSet.remove(labels[i])
            labels[i] = tmpSet[random.randint(0, 1)]
    return labels


def random_forest(train, test):

    clf = RandomForestClassifier(n_jobs=2, random_state=0)  # Initializes RF classifier, n_jobs: parralelizes
    clf.fit(train[train.columns[:4]], train['Names'])  # trains the classifier

    return clf.score(test[test.columns[:4]],test['Names'], sample_weight=None)

def random_forest_mislabeling():
    p_vec = []
    p = 0
    p_increment = 0.025
    n_iter = 40
    n_mean = 30
    success_vec = np.zeros(n_iter)
    for i in range(0, n_iter):
        p_vec.append(p)
        success_tmp = 0
        for j in range(0, n_mean):
            iris = datasets.load_iris()
            train = pd.DataFrame(iris.data[:, :4])
            train['Names'] = iris.target
            test = train.sample(25)
            train = train.drop(test.index)


            labels = []
            for e in list(train['Names']):
                labels.append(e)

            labels = mislabel(labels, p)

            #replaces old labels in training data
            del train['Names']
            train.loc[:, 'Names'] = labels

            success_tmp += (random_forest(train, test))

        success_vec[i] = (success_tmp/n_mean)
        p += p_increment
        print('Done with iteration ' + str(i + 1) + ' of ' + str(n_iter) + '.')
    return p_vec, success_vec
'''
x = []
for i in range(0,10):
    iris = datasets.load_iris()
    train = pd.DataFrame(iris.data[:, :4])
    train['Names'] = iris.target
    test = train.sample(30)
    train = train.drop(test.index)
    x.append(random_forest(train,test))
'''
p_vec, success_vec = random_forest_mislabeling()
plt.plot(p_vec,success_vec)
plt.show()
print()