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

'''
Simulates accuracy in classification model RF or KNN as a function of mislabeling

@:param model: determines what model is to be used if == 1 --> random forest // if == 2 --> knn
@:param modelparam: if model is RF, this is #trees; if KNN; this is K
@:param mislabeltype: adversarial or random
@:param p_increment: mislabeling probability step size
@:param n_iter: number of iterations
@:param n_mean: number of runs for one iteration over which mean is taken

@:return p_vec: Mislabeling probabilities in a vector
@:return accuracy_vec: Model accuracies corresponding to the probabilities

'''
def model_mislabeling(model, modelparam, mislabeltype, p_increment, n_iter, n_mean):
    p_vec = []
    p = 0
    accuracy_vec = np.zeros(n_iter)
    for i in range(0, n_iter):
        p_vec.append(p)
        accuracy = 0
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

            accuracy += (clf(train, test, model, modelparam))

            accuracy_vec[i] = (accuracy/n_mean)
        p += p_increment
        print('Done with iteration ' + str(i + 1) + ' of ' + str(n_iter) + '.')
    return p_vec, accuracy_vec


p_vec, accuracy_vec = model_mislabeling(model=2, modelparam = 10, mislabeltype = 'random',
                                       p_increment = 0.025, n_iter = 40, n_mean = 10)

plt.plot(p_vec, accuracy_vec, 'b-.', lw=2, label='KNN Accuracy (K = 10)')
plt.title('KNN vs RF; random mislabeling')
plt.legend(loc='upper right')
plt.show()
print()