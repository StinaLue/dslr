#!/usr/bin/env python

import pandas
import matplotlib.pyplot as plt
import numpy as np

"""
Histogram which shows which Hogwarts course has a homogeneous score distribution between all four houses
"""

df = pandas.read_csv("./dataset_train.csv")