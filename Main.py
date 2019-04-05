import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


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

