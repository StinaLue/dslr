#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools

"""
Scatter plot which shows which Hogwarts course has a homogeneous score distribution between all four houses

What are the two features that are similar ?
"""

df = pd.read_csv("./dataset_train.csv")

color_dict = dict({'Hufflepuff':'gold',
                  'Slytherin':'lime',
                  'Ravenclaw': 'blue',
                  'Gryffindor': 'red'})

nrows = 6 #78 / 13
ncolumns = 13 #78 / 6
fig, axes = plt.subplots(nrows, ncolumns)

index = 0

for ft_names in itertools.combinations(df.columns[6:], 2):
    sns.scatterplot(ax=axes[int(index / ncolumns), int(index % ncolumns)], data=df, x=ft_names[0], y=ft_names[1], hue="Hogwarts House", palette=color_dict, legend=False)
    index += 1

#plt.savefig("scatter_plot.png")
plt.show()
