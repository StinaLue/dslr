#!/usr/bin/env python3

from ft_MachineLearning.LogisticRegression import LogisticRegression_Model
from argparse import ArgumentParser
import os
import pandas as pd

def verify_csv(filename):
    """
    very basic check for the CSV file --> it has to exist, and end in .csv
    """
    if not os.path.exists(filename):
        print("The file %s does not exist!" % filename)
        exit()
    if not filename.endswith('.csv'):
        print("The file %s is not a csv file!" % filename)
        exit()

def read_csv(filename):
    verify_csv(filename)
    dataframe = pd.read_csv(filename)
    return dataframe

def preprocess_dataframe(df):
    df = df.fillna(df.mean())
    return df

def parse_arguments():  
    parser = ArgumentParser()
    parser.add_argument("-f",
        dest="filename",
        help="Filename of the dataset CSV file",
        required=True)
    parser.add_argument("-v", "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Give a verbose output of the linear regression training")
    parser.add_argument("--classname",
        dest="classname",
        default=None,
        help="Name of the class category that we try to predict")
    parser.add_argument("--valuepredict",
        dest="valuepredict",
        default=None,
        help="Name or value, defines what in the dataset is going to be a 1, and the rest will be a 0")
    parser.add_argument("--features",
        dest="features",
        default=None,
        help="features to use for training in the given file, shape is 'featurename1,featurename2...'")
    return parser.parse_args()


def main():
    args = parse_arguments()
    #(PROCESS ARGS)
    hogwarts_df = read_csv(args.filename)
    hogwarts_df = preprocess_dataframe(hogwarts_df)
    features_selected = ["Astronomy", "Herbology", "Charms", "Ancient Runes"]
    Gryffindor_Model = LogisticRegression_Model(hogwarts_df, "Gryffindor", "Hogwarts House", features_selected)
    #print(Gryffindor_Model.features_matrix)
    Gryffindor_Model.train(Gryffindor_Model.scaled_features_matrix, Gryffindor_Model.classes, 1e-3, 50, 10, 1e-8)
    """
    if args.features is None or args.classname is None or args.valuepredict is None:
        logmodel = LinearRegression_Model(args.filename, verbose=args.verbose)
    else:
        try:
            features = args.features.split(",")
            data = LinearRegression_Model(args.filename, xfeature=xfeature, yfeature=yfeature, verbose=args.verbose)
        except:
            print("Something went wrong with the features you gave.")
            exit()
    """

main()