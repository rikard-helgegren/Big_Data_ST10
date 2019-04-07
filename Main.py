import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
pd.options.mode.chained_assignment = None

def successRate(pred, true):
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


def randomForest(train, test):
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
    return successRate(y_pred, y_true)


#reads iris data
df = pd.read_csv('iris.csv')



success_vec = []
p_vec = []
p = 0
p_increment = 0.01
n_iter = 20
n_mean = 10

for i in range(0, n_iter):
    p_vec.append(p)
    success_tmp = 0
    for j in range(0, n_mean):
        # assign wether observations should be used for training or not.
        df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75  # ~25% validerigsdata
        train, test = df[df['is_train'] == True], df[df['is_train'] == False]

        labels = []
        for e in list(train['Names']):
            labels.append(e)

        #mislabels randomly
        labels = mislabel(labels, p)

        #replaces old labels in training data
        del train['Names']
        train.loc[:, 'Names'] = labels

        success_tmp += (randomForest(train, test))

    success_vec.append(success_tmp/n_mean)
    p += p_increment
    print('Done with iteration ' + str(i + 1) + ' of ' + str(n_iter) + '.')



plt.plot(p_vec, success_vec)
plt.show()
print()


