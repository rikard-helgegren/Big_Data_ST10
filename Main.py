import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pointbiserialr
from pandas.plotting import parallel_coordinates
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
import math
import random
from math import isnan
from math import isnan
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial.distance as ssd
from scipy.spatial import distance_matrix


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


df_misl = pd.read_csv(r'C:\Users\arvid\Desktop\Skola\Skolår 3\Kandidatarbete\iris.csv')
df_misl['Names'] = mislabel(df_misl, .5)

df_true = pd.read_csv(r'C:\Users\arvid\Desktop\Skola\Skolår 3\Kandidatarbete\iris.csv')

print()

