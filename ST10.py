'''
Authors:
        Arvid Wartenberg
        Rikard Helgegren
        Lina Hammargren

Group: ST10

'''


#Module imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


'''
     Mislabels a fraction p of the labels

    @:params: labels, the set of all labels to be mislabeled
    @:params: p, the fraction of the whole set that should be mislabeled

    @:return: labels, mislabeled labels
'''
def mislabel(labels, p):

    #set of unique elements in labels
    labelSet = list(set(labels))

    #loops over all labels
    for i in range(0, len(labels)):

        #randomizes wether index i should be mislabeled
        if random.uniform(0, 1) < p:

            #tmpSet is made, containing all labels except the true label
            tmpSet = []
            for j in range(0,len(labelSet)):
                tmpSet.append(labelSet[j])
            tmpSet.remove(labels[i])

            #new false label is set for index i
            labels[i] = tmpSet[random.randint(0, 1)]

    #return statement
    return labels

'''
    Mislabels a fraction p of the labels 'versicolor' and 'virginica' to
    class 'setosa'

    @:params: labels, the set of all labels to be mislabeled
    @:params: p, the fraction of the whole set that should be mislabeled

    @:return: labels, mislabeled labels
'''
def adversarial_mislabel(labels, p):

    #loops over all labels
    for i in range(0, len(labels)):

        #randomizes wether label at index i should be mislabeled
        if random.uniform(0, 1) < p:

            #label at index i is set to 0
            labels[i] = 0

    #return statement
    return labels



'''
Trains a classifier on training data and scores it on test data (RF or KNN)

@:param train: training data
@:param test: test data
@:param model: determines what model is to be used if == 1 --> random forest // if == 2 --> knn
@:param modelparam: if model is RF, this is #trees; if KNN; this is K

@:returns: Model accuracy when predicting on test data
'''
def clf(train, test, model, modelparam):

    #initializes classifier
    if model == 1:
        clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=modelparam)  # Initializes RF classifier, n_jobs: parralelizes
    elif model == 2:
        clf = KNeighborsClassifier(n_neighbors=modelparam)

    #trains classifier
    clf.fit(train[train.columns[:4]], train['Names'])  # trains the classifier

    #returns accuracy
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

    #initializing p_vec and accuracy_vec
    p_vec = []
    p = 0
    accuracy_vec = np.zeros(n_iter)

    #Runs iterations over probabilities
    for i in range(0, n_iter):
        p_vec.append(p)
        accuracy = 0

        #trains n_mean classifiers and takes mean of accuracy
        for j in range(0, n_mean):

            #loads iris data and divides into training and test sets
            iris = datasets.load_iris()
            train = pd.DataFrame(iris.data[:, :4])
            train['Names'] = iris.target
            test = train.sample(25)
            train = train.drop(test.index)


            #mislabels in training set
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

        #appends current average accuracy to accuracy vec
        accuracy_vec[i] = (accuracy/n_mean)

        #increases mislabeling probability
        p += p_increment

        print('Done with iteration ' + str(i + 1) + ' of ' + str(n_iter) + '.')

    #return statement
    return p_vec, accuracy_vec



#gets p_vec, accuracy_vec for chosen model and params..
p_vec, accuracy_vec = model_mislabeling(model=2, modelparam = 10, mislabeltype = 'random',
                                       p_increment = 0.025, n_iter = 40, n_mean = 10)


#plots results
plt.plot(p_vec, accuracy_vec, 'b-.', lw=2, label='KNN Accuracy (K = 10)')
plt.title('KNN vs RF; random mislabeling')
plt.legend(loc='upper right')
plt.show()