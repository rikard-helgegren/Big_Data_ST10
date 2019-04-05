import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier


def mislabel(df, p):
    labels = df['Names']
    labelSet = list(set(labels))
    for i in range(0, len(labels)):
        if random.uniform(0, 1) < p:
            tmpSet = []
            for j in range(0,len(labelSet)):
                tmpSet.append(labelSet[j])
            tmpSet.remove(labels[i])
            labels[i] = tmpSet[random.randint(0, 1)]
    return labels


df = pd.read_csv('iris.csv')
df['Names'] = mislabel(df, .5) #mislabels dataframe
df_true = pd.read_csv('iris.csv') #df with true labels


#assign wether observations should be used for training or not.
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75 #~25% validerigsdata
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Show the number of observations for the test and training dataframes'
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Create a list of the feature column's names
features = df.columns[:4]

# train['species'] contains the actual species names. Before we can use it,
# we need to convert each species name into a digit. So, in this case there
# are three species, which have been coded as 0, 1, or 2.
y = pd.factorize(train['species'])[0]

clf = RandomForestClassifier(n_jobs=2, random_state=0) #n_jobs: parralelizes



print()

