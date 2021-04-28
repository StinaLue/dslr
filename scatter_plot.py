#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
Histogram which shows which Hogwarts course has a homogeneous score distribution between all four houses
"""

df = pd.read_csv("./dataset_train.csv")

color_dict = dict({'Hufflepuff':'gold',
                  'Slytherin':'lime',
                  'Ravenclaw': 'blue',
                  'Gryffindor': 'red'})
#sns.set_theme()
nrows = 3
ncolumns = 5
fig, axes = plt.subplots(nrows, ncolumns, figsize=(18, 10), sharey=True, squeeze=True)
for i,feature in enumerate(df.columns[6:]):
    sns.scatterplot(ax=axes[int(i / ncolumns), int(i % ncolumns)], data=df, hue="Hogwarts House", y="Flying", x=feature, palette=color_dict)
plt.show()