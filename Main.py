import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
pd.options.mode.chained_assignment = None

def success_rate(pred, true):
    n = len(pred)
    n_right = 0
    for i in range(0, n):
        if pred[i] == true[i]:
            n_right += 1
    return n_right/n


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
    labelSet = list(set(labels))
    setosa = 0
    versicolor=0
    virginica=0

    
    for j in range(0,len(labels)):
        if labels[j] == 'Iris-setosa':
            setosa += 1
        elif labels[j] == 'Iris-versicolor':
            versicolor += 1
        elif labels[j] == 'Iris-virginica':
            virginica += 1
    frac = (setosa)/(virginica+setosa+versicolor)

    for i in range(0, len(labels)):
        if random.uniform(0, 1) < p*frac:
            tmpSet = []

            if labels[i] != 'Iris-setosa':
                labels[i] = 'Iris-setosa'

    return labels

def random_forest(train, test):
    # Show the number of observations for the test and training dataframes'
    #print('Number of observations in the training data:', len(train))
    #print('Number of observations in the test data:', len(test))

    # Create a list of the feature column's names
    features = df.columns[:4]

    # train['species'] contains the actual species names. Before we can use it,
    # we need to convert each species name into a digit. So, in this case there
    # are three species, which have been coded as 0, 1, or 2.
    y = pd.factorize(train['Names'])[0]

    clf = RandomForestClassifier(n_jobs=2, random_state=0)  # Initializes RF classifier, n_jobs: parralelizes
    clf.fit(train[features], y)  # trains the classifier

    # predicts with the trained model
    y_pred = clf.predict(test[features])
    y_true = pd.factorize(test['Names'])[0]# These are the true labels
    #pd.crosstab(test['Names'], y_pred, rownames=['Actual Species'], colnames=['Predicted Species']) #confusion matrix
    return success_rate(y_pred, y_true)


def knn(k, train, test): # gör 3d!

    # Create a list of the feature column's names
    features = df.columns[:4]

    # train['species'] contains the actual species names. Before we can use it,
    # we need to convert each species name into a digit. So, in this case there
    # are three species, which have been coded as 0, 1, or 2.
    y = pd.factorize(train['Names'])[0]

    clf = KNeighborsClassifier(n_neighbors=k)  # Initializes KNN classifier, n_jobs: parralelizes
    clf.fit(train[features], y)  # trains the classifier

    # predicts with the trained model
    y_pred = clf.predict(test[features])
    y_true = pd.factorize(test['Names'])[0]# These are the true labelss

    return success_rate(y_pred, y_true)




def random_forest_mislabeling(df, type):
    p_vec = []
    p = 0
    p_increment = 0.02
    n_iter = 40
    n_mean = 10
    success_vec = np.zeros(n_iter)
    for i in range(0, n_iter):
        p_vec.append(p)
        success_tmp = 0
        for j in range(0, n_mean):
            #TODO:explain to rikard how this section works.
            # assign wether observations should be used for training or not. 
            df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75  # ~25% validerigsdata
            train, test = df[df['is_train'] == True], df[df['is_train'] == False]

            labels = []
            for e in list(train['Names']):
                labels.append(e)

            #mislabels randomly
            if type is 'random':
                labels = mislabel(labels, p)
            else:
                labels = adversarial_mislabel(labels, p)

            #replaces old labels in training data
            del train['Names']
            train.loc[:, 'Names'] = labels

            success_tmp += (random_forest(train, test))

        success_vec[i] = (success_tmp/n_mean)
        p += p_increment
        print('Done with iteration ' + str(i + 1) + ' of ' + str(n_iter) + '.')
    return p_vec, success_vec

def knn_mislabeling(df, type):
    success_vec = []
    p_vec = []
    p_increment = 0.02
    n_iter = 40
    k_iter = 1
    n_mean = 40
    p_mat = np.zeros([k_iter, n_iter])
    k_mat = np.zeros([k_iter, n_iter])
    success_map = np.zeros([k_iter, n_iter])
    for k in range(0, k_iter):
        p = 0
        for i in range(0, n_iter):
            k_act = 2*k+1
            p_vec.append(p)
            p_mat[k,i] = p
            k_mat[k, i] = k_act
            success_tmp = 0
            for j in range(0, n_mean):
                # assign wether observations should be used for training or not.
                df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75  # ~25% validerigsdata
                train, test = df[df['is_train'] == True], df[df['is_train'] == False]

                labels = []
                for e in list(train['Names']):
                    labels.append(e)

                #mislabels randomly
                if type is 'random':
                    labels = mislabel(labels, p)
                else:
                    labels = adversarial_mislabel(labels, p)

                #replaces old labels in training data
                del train['Names']
                train.loc[:, 'Names'] = labels

                success_tmp += (knn(k_act,train, test))

            success_map[k,i] = success_tmp/n_mean
            success_vec.append(success_tmp/n_mean)
            p += p_increment
            print('Done with iteration ' + str(i + 1) + ' of ' + str(n_iter) + '.')
    return success_map, p_mat, k_mat

#reads iris data
df = pd.read_csv('iris.csv')

#success_map, p_mat, k_mat = knn_mislabeling(df, 'random')

p_vec, success_vec = random_forest_mislabeling(df, 'random')
p_vec_, succes_mat, k_mat = knn_mislabeling(df, 'random')


plt.plot(p_vec, success_vec)
plt.plot(succes_mat[0,:], p_vec_[0,:])
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(Z = success_map, X = p_mat, Y = k_mat, cmap = 'magma')
plt.show()
print()


