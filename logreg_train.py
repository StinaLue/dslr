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
    return parser.parse_args()


def main():
    args = parse_arguments()
    #(PROCESS ARGS)
    hogwarts_df = read_csv(args.filename)
    hogwarts_df = preprocess_dataframe(hogwarts_df)
    features_selected = ["Astronomy", "Herbology", "Charms", "Ancient Runes"]
    Gryffindor_Model = LogisticRegression_Model(hogwarts_df, features_selected, "Hogwarts House", "Gryffindor")
    Gryffindor_Model.train(Gryffindor_Model.scaled_features_matrix, Gryffindor_Model.real_labels, 1e-1, 50, 15000, 1e-4)
    Gryffindor_Model.save_weights_csv("gryffindor_weights.csv")

    Hufflepuff_Model = LogisticRegression_Model(hogwarts_df, features_selected, "Hogwarts House", "Hufflepuff")
    Hufflepuff_Model.train(Hufflepuff_Model.scaled_features_matrix, Hufflepuff_Model.real_labels, 1e-1, 50, 15000, 1e-4)
    Hufflepuff_Model.save_weights_csv("hufflepuff_weights.csv")

    Ravenclaw_Model = LogisticRegression_Model(hogwarts_df, features_selected, "Hogwarts House", "Ravenclaw")
    Ravenclaw_Model.train(Ravenclaw_Model.scaled_features_matrix, Ravenclaw_Model.real_labels, 1e-1, 50, 15000, 1e-4)
    Ravenclaw_Model.save_weights_csv("ravenclaw_weights.csv")
    
    Slytherin_Model = LogisticRegression_Model(hogwarts_df, features_selected, "Hogwarts House", "Slytherin")
    Slytherin_Model.train(Slytherin_Model.scaled_features_matrix, Slytherin_Model.real_labels, 1e-1, 50, 15000, 1e-4)
    Slytherin_Model.save_weights_csv("slytherin_weights.csv")

    filenames = ['gryffindor_weights.csv', 'hufflepuff_weights.csv', 'ravenclaw_weights.csv', 'slytherin_weights.csv']
    with open('./weights.csv', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                outfile.write(infile.read())

main()
