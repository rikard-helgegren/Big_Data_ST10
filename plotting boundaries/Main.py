from read_CSV       import read_CSV
from split_data     import split_data_list
from misslabel_data import misslabel_data_list
from convert_data   import convert_data

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.ensemble           import RandomForestClassifier
from sklearn.model_selection    import train_test_split
from sklearn.model_selection    import cross_val_score
from sklearn.pipeline           import make_pipeline


def plot_decision_boundary(clf, X, Y, cmap=plt.cm.RdYlBu):
    h = 0.02
    x_min, x_max = X[:,0].min() - 10*h, X[:,0].max() + 10*h
    y_min, y_max = X[:,1].min() - 10*h, X[:,1].max() + 10*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(5,5))
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolors='k');


def randome_forest_classifier_crossval(train, nr_trees=20, max_depth=2):
    folds = 5

    # Separate labels from data
    data = [row[:-1] for row in train]
    labels = [row[-1] for row in train]

    clf = RandomForestClassifier(n_estimators=nr_trees, max_depth=max_depth,
                                 random_state=0)
    scores = cross_val_score(clf, data, labels, cv=folds)
    mean_score = sum(scores)/len(scores)

    return mean_score


def randome_forest_classifier_plot(train, nr_trees=20, max_depth=2):
    
    #make list to np.array
    train = convert_data(train)

    # Separate labels from data
    data = train[:, :4]
    labels = train[:, -1]

    #define which parameters to plot
    data_to_plot = data[:, [2, 3]]  


    clf = RandomForestClassifier(n_estimators=nr_trees, max_depth=max_depth,
                                 random_state=0)
    

    clf.fit(data_to_plot, labels)

    scores = clf.score(data_to_plot, labels)

    plot_decision_boundary(clf, data_to_plot, labels)
    plt.draw()

    return scores



data = read_CSV('iris.csv')

#remove param descriptions
data = data[1:]

[train, validation]= split_data_list(data, 0.2) 

# Need deep coppy in order not to change in list
misslabel_data=misslabel_data_list(copy.deepcopy(train), 0.2)


print("Score cross validation:", randome_forest_classifier_crossval(data))
print("Score cross validation misslabeled:", randome_forest_classifier_crossval(misslabel_data))

print("Score fit:", randome_forest_classifier_plot(data))
print("Score fit misslabeled:", randome_forest_classifier_plot(misslabel_data))


plt.show()


