#!/usr/bin/env python

import pandas
import matplotlib.pyplot as plt
import numpy as np
from numbers import Number
from pandas.api.types import is_numeric_dtype
import math

class FeatureData():
    """
    Takes a numerical pandas Series and computes several attributes related to its data

    Attributes
    ------------
    count : nb of entries
    mean : average value from all entries
    std : standard deviation (amount of variation or dispersion in the values)
    min : minimum value in all entries
    per25/50/75 : specific values at 25%/50%/75%
    max : maximum value in all entries
    name : name of the feature (Series.name)
    feature : actual values of the features (Series)
    """
    def __init__(self, feature):
        """
        Parameters
        --------------
        feature = pandas Series (sorted or not)
        """
        self.count = 0
        self.mean = 0
        self.std = 0
        self.min = 0
        self.per25 = 0
        self.per50 = 0
        self.per75 = 0
        self.max = 0
        self.name = feature.name
        self.feature = feature

        self.__clean_empty_feature()
        self.__sort_feature()
        self.__count_feature()
        self.__min_feature()
        self.__per25_feature()
        self.__per50_feature()
        self.__per75_feature()
        self.__max_feature()
        self.__mean_feature()
        self.__std_feature()

    def get_data_dict(self):
        return({
            "name": self.name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "25%": self.per25,
            "50%": self.per50,
            "75%": self.per75,
            "max": self.max
        })
        
    def get_data_list(self):
        return([
            self.name,
            self.count,
            self.mean,
            self.std,
            self.min,
            self.per25,
            self.per50,
            self.per75,
            self.max
        ])

    def __sort_feature(self):
        self.feature = self.feature.sort_values()
    
    def __clean_empty_feature(self):
        self.feature = self.feature.dropna()
    
    def __count_feature(self):
        for row in self.feature:
            self.count += 1
        self.count = float(self.count)
    
    def __min_feature(self):
        self.min = self.feature[0]
    
    def __max_feature(self):
        self.max = self.feature[self.feature.last_valid_index()]

    def __std_feature(self):
        """
        https://stackoverflow.com/questions/25695986/pandas-why-pandas-series-std-is-different-from-numpy-std/25696030
        """
        deviations = [(x - self.mean) ** 2 for x in self.feature]
        variance = sum(deviations) / (self.count - 1)
        self.std = variance ** 0.5
    
    def __mean_feature(self):
        self.mean = self.feature.sum() / self.count

    def __per25_feature(self):
        """
        https://stackoverflow.com/questions/2374640/how-do-i-calculate-percentiles-with-python-numpy
        """
        index = 0.25 * (self.count - 1)
        c = math.ceil(index)
        f = math.floor(index)

        if c == f:
            self.per25 = self.feature.values[int(index)]
        else:
            d0 = self.feature.values[int(f)] * (c - index)
            d1 = self.feature.values[int(c)] * (index - f)
            self.per25 = d0 + d1

    def __per50_feature(self):
        """
        https://stackoverflow.com/questions/2374640/how-do-i-calculate-percentiles-with-python-numpy
        """
        index = 0.50 * (self.count - 1)
        c = math.ceil(index)
        f = math.floor(index)

        if c == f:
            self.per50 = self.feature.values[int(index)]
        else:
            d0 = self.feature.values[int(f)] * (c - index)
            d1 = self.feature.values[int(c)] * (index - f)
            self.per50 = d0 + d1

    def __per75_feature(self):
        """
        https://stackoverflow.com/questions/2374640/how-do-i-calculate-percentiles-with-python-numpy
        """
        index = 0.75 * (self.count - 1)
        c = math.ceil(index)
        f = math.floor(index)

        if c == f:
            self.per75 = self.feature.values[int(index)]
        else:
            d0 = self.feature.values[int(f)] * (c - index)
            d1 = self.feature.values[int(c)] * (index - f)
            self.per75 = d0 + d1
    
def filter_dataframe(dataframe):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    filtered_dataframe = dataframe.select_dtypes(include=numerics)
    filtered_dataframe.pop("Index")
    return (filtered_dataframe)

def describe(dataframe):
    filtered_dataframe = filter_dataframe(dataframe)
    sorted_columns = []
    for column_name in filtered_dataframe:
        sorted_columns.append(filtered_dataframe[column_name].sort_values())

df = pandas.read_csv("./dataset_train.csv")
filtered_df = filter_dataframe(df)
lst = []
lst.append([
            "name",
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max"
            ])

for feature in filtered_df:
    lst.append(FeatureData(filtered_df[feature]).get_data_list())
for indexes,ft1,ft2,ft3,ft4,ft5,ft6,ft7,ft8,ft9,ft10,ft11,ft12,ft13 in zip(*lst):
    if (type(ft1) == str):
        print("{:<5}{:>14.10}{:>14.10}{:>14.10}{:>14.10}{:>14.10}{:>14.10}{:>14.10}{:>14.10}{:>14.10}{:>14.10}{:>14.10}{:>14.10}{:>14.10}".format("",ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12, ft13))
    else:
        print("{:<5}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}".format(indexes, ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9, ft10, ft11, ft12, ft13))

#for i in range(len(lst)):
    #print(lst[i].min, end="\t")
#    print("{:.5f}".format(lst[i].min), end="\t")
#print(df.describe())
#test.__sort_feature()
#test.__clean_empty_feature()