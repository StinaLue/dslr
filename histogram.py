#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
Histogram which shows which Hogwarts course has a homogeneous score distribution between all four houses
"""

df = pd.read_csv("./dataset_train.csv")

#hufflepuffs = df[df['Hogwarts House' == 'Hufflepuff']]

color_dict = dict({'Hufflepuff':'gold',
                  'Slytherin':'lime',
                  'Ravenclaw': 'blue',
                  'Gryffindor': 'red'})
#sns.set_theme()
fig, axes = plt.subplots(3, 5, figsize=(18, 10), sharey=True)
sns.histplot(ax=axes[0, 0], data=df, x="Herbology", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[0, 1], data=df, x="Flying", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[0, 2], data=df, x="Arithmancy", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[0, 3], data=df, x="Astronomy", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[1, 0], data=df, x="Defense Against the Dark Arts", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[1, 1], data=df, x="Divination", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[1, 2], data=df, x="Muggle Studies", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[1, 3], data=df, x="Ancient Runes", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[2, 0], data=df, x="History of Magic", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[2, 1], data=df, x="Transfiguration", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[2, 2], data=df, x="Charms", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[2, 3], data=df, x="Potions", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True
sns.histplot(ax=axes[1, 4], data=df, x="Care of Magical Creatures", hue="Hogwarts House", palette=color_dict, element="step", bins=15)#,kde=True

plt.show()