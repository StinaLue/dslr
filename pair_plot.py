#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
Pairplot which shows which Hogwarts courses are the most similar
"""

df = pd.read_csv("./dataset_train.csv")

color_dict = dict({'Hufflepuff':'gold',
                  'Slytherin':'lime',
                  'Ravenclaw': 'blue',
                  'Gryffindor': 'red'})
sns.set_theme()
g = sns.pairplot(df.drop("Index", axis=1), hue="Hogwarts House", palette=color_dict, dropna=True)
g.savefig("pairplot.png")
plt.show()
