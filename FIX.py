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


def adversarial_mislabel(labels, p):
    for i in range(0, len(labels)):
        if random.uniform(0, 1) < p:
            labels[i] = 0

    return labels


def clf(train, test, model, modelparam):
    if model == 1:
        clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=modelparam)  # Initializes RF classifier, n_jobs: parralelizes
    elif model == 2:
        clf = KNeighborsClassifier(n_neighbors=modelparam)


    clf.fit(train[train.columns[:4]], train['Names'])  # trains the classifier

    return clf.score(test[test.columns[:4]],test['Names'], sample_weight=None)


def model_mislabeling(model, modelparam, mislabeltype, p_increment, n_iter, n_mean):
    p_vec = []
    p = 0
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
            if mislabeltype == 'random':
                labels = mislabel(labels, p)
            elif mislabeltype == 'adversarial':
                labels = adversarial_mislabel(labels, p)


            #replaces old labels in training data
            del train['Names']
            train.loc[:, 'Names'] = labels

            success_tmp += (clf(train, test, model, modelparam))

        success_vec[i] = (success_tmp/n_mean)
        p += p_increment
        print('Done with iteration ' + str(i + 1) + ' of ' + str(n_iter) + '.')
    return p_vec, success_vec

#model: 1 -> RF, 2 -> KNN, modelparam = (#trees for RF// K for KNN), TODO document
p_vec, success_vec = model_mislabeling(model=2, modelparam = 10, mislabeltype = 'adversarial',
                                       p_increment = 0.025, n_iter = 40, n_mean = 10)
p_vec2, success_vec2 = model_mislabeling(model=1, modelparam = 100, mislabeltype = 'adversarial',
                                       p_increment = 0.025, n_iter = 40, n_mean = 4)

plt.plot(p_vec, success_vec, 'b-.', lw=2, label='KNN Accuracy (K = 10)')
plt.plot(p_vec2, success_vec2, '--', lw=2, label='RF Accuracy (100 trees)')
plt.title('KNN vs RF; adversarial mislabeling')
plt.legend(loc='upper right')
plt.show()
print()