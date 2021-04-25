#!/usr/bin/env python

import pandas
import matplotlib.pyplot as plt
import numpy as np
from numbers import Number
from pandas.api.types import is_numeric_dtype

class Feature:
    def __init__(self, column):
        self.count
    
    def filter_dataframe(self, dataframe):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        filtered_dataframe = dataframe.select_dtypes(include=numerics)
        filtered_dataframe.pop("Index")
        print(filtered_dataframe)
        return (filtered_dataframe)
    
    def calculation_count(self, column):
        count = 0
        for value in column:
                count += 1
        amount_nan_values = column.isnull().sum()
        self.count = count - amount_nan_values

def describe(self, dataframe):
    print(dataframe.describe())
    filtered_dataframe = filter_dataframe(dataframe)
    for column in filtered_dataframe:
        sorted_column = filtered_dataframe.sort_values(column)
        calculation_count(sorted_column[column])

dataframe = pandas.read_csv("./dataset_train.csv")
features = []
for column in dataframe:
    features = 
