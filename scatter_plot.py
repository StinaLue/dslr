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

#def hide_current_axis(*args, **kwds):
#    plt.gca().set_visible(False)
#sns.set_theme()

#g = sns.PairGrid(df.drop("Index", axis=1), hue="Hogwarts House", palette=color_dict, dropna=True, corner=True)
#g.map_lower(sns.scatterplot, alpha=0.3, edgecolor='none')
#g.map_diag(hide_current_axis)
#g.savefig("scatter_plot.png")

# n! / r!(n - r)!
#(13)! / ((2)! * (11)!) --> 78 possibilities

nrows = 6 #78 / 13
ncolumns = 13 #78 / 6
#fig, axes = plt.subplots(nrows, ncolumns)#, figsize=(18, 10))#, sharey=True)#, squeeze=True)
fig, axes = plt.subplots(nrows, ncolumns)#, constrained_layout=True)# figsize=(18, 10))#, sharey=True)#, squeeze=True)
#for i,feature in enumerate(df.columns[6:]):
#   sns.scatterplot(ax=axes[int(i / ncolumns), int(i % ncolumns)], data=df, x="Arithmancy", y="Flying", hue="Hogwarts House", palette=color_dict, legend=False)
    #fig.delaxes(axes.flatten()[i])
#for i, ft_name in enumerate(df.columns[6:]):
#    for j, ft_name2 in enumerate(df.columns[6:]):
#        sns.scatterplot(ax=axes[i, j], data=df, x=ft_name, y=ft_name2, hue="Hogwarts House", palette=color_dict, legend=False)

index = 0

for ft_names in itertools.combinations(df.columns[6:], 2):
    sns.scatterplot(ax=axes[int(index / ncolumns), int(index % ncolumns)], data=df, x=ft_names[0], y=ft_names[1], hue="Hogwarts House", palette=color_dict, legend=False)
    index += 1

#fig.tight_layout()
#plt.tight_layout()
#plt.savefig("scatter_plot.png")
plt.show()
