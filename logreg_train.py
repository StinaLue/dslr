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
    Gryffindor_Model = LogisticRegression_Model(hogwarts_df, features_selected, "Hogwarts House", "Gryffindor")
    Gryffindor_Model.train(Gryffindor_Model.scaled_features_matrix, Gryffindor_Model.classes, 1e-1, 50, 15000, 1e-4)
    Gryffindor_Model.save_weights_csv("gryffindor_weights.csv")

    Hufflepuff_Model = LogisticRegression_Model(hogwarts_df, features_selected, "Hogwarts House", "Hufflepuff")
    Hufflepuff_Model.train(Hufflepuff_Model.scaled_features_matrix, Hufflepuff_Model.classes, 1e-1, 50, 15000, 1e-4)
    Hufflepuff_Model.save_weights_csv("hufflepuff_weights.csv")

    Ravenclaw_Model = LogisticRegression_Model(hogwarts_df, features_selected, "Hogwarts House", "Ravenclaw")
    Ravenclaw_Model.train(Ravenclaw_Model.scaled_features_matrix, Ravenclaw_Model.classes, 1e-1, 50, 15000, 1e-4)
    Ravenclaw_Model.save_weights_csv("ravenclaw_weights.csv")
    
    Slytherin_Model = LogisticRegression_Model(hogwarts_df, features_selected, "Hogwarts House", "Slytherin")
    Slytherin_Model.train(Slytherin_Model.scaled_features_matrix, Slytherin_Model.classes, 1e-1, 50, 15000, 1e-4)
    Slytherin_Model.save_weights_csv("slytherin_weights.csv")
"""
    #testdata = pd.read_csv("dataset_train.csv")
    testdata = pd.read_csv("dataset_test.csv")
    testdata = preprocess_dataframe(testdata)
    scaled_test_data = (testdata - testdata.mean()) / testdata.std()
    scaled_features = scaled_test_data[features_selected].to_numpy()
    #class_real = testdata["Hogwarts House"]
    firstnames = testdata["First Name"]
    #real_classes = (testdata["Hogwarts House"] == "Gryffindor").astype(int).to_numpy()
    #print(Gryffindor_Model.features_coeffs)
    probabilites = Gryffindor_Model.predict_proba(scaled_features, Gryffindor_Model.features_coeffs, Gryffindor_Model.intercept)
    print(Gryffindor_Model.features_coeffs)
    print(Gryffindor_Model.intercept)
    #real_classes_s = (testdata["Hogwarts House"] == "Slytherin").astype(int).to_numpy()
    probabilites_s = Slytherin_Model.predict_proba(scaled_features, Slytherin_Model.features_coeffs, Slytherin_Model.intercept)

    probabilites_h = Hufflepuff_Model.predict_proba(scaled_features, Hufflepuff_Model.features_coeffs, Hufflepuff_Model.intercept)

    probabilites_r = Ravenclaw_Model.predict_proba(scaled_features, Ravenclaw_Model.features_coeffs, Ravenclaw_Model.intercept)
    
    for firstname, proba_G, proba_S, proba_H, proba_R in zip(firstnames, probabilites, probabilites_s, probabilites_h, probabilites_r):
        print("Hmmmm, " + str(firstname) + "is interesting... ", end="")
        proba_tab = [proba_G, proba_S, proba_H, proba_R]
        max_proba = max(proba_tab)
        if (max_proba == proba_G):
            print("Sorting Hat says : 'GRYFFINDOR !!!'")
        elif (max_proba == proba_S):
            print("Sorting Hat says : 'SLYTHERIN !!!'")
        elif (max_proba == proba_H):
            print("Sorting Hat says : 'HUFFLEPUFF !!!'")
        elif (max_proba == proba_R):
            print("Sorting Hat says : 'RAVENCLAW !!!'")
        else:
            print("Sorting Hat says : 'am confoos'")
    """
    """
    for realclass, proba_G, proba_S, proba_H, proba_R in zip(class_real, probabilites, probabilites_s, probabilites_h, probabilites_r):
        print("True is " + str(realclass) + " ", end="")
        proba_tab = [proba_G, proba_S, proba_H, proba_R]
        max_proba = max(proba_tab)
        if (max_proba == proba_G):
            print("Sorting Hat says : 'GRYFFINDOR !!!'")
        elif (max_proba == proba_S):
            print("Sorting Hat says : 'SLYTHERIN !!!'")
        elif (max_proba == proba_H):
            print("Sorting Hat says : 'HUFFLEPUFF !!!'")
        elif (max_proba == proba_R):
            print("Sorting Hat says : 'RAVENCLAW !!!'")
        else:
            print("Sorting Hat says : 'am confoos'")
    """
    """
    for realclass, probability in zip(real_classes, probabilites):
        print(realclass, end="")
        print(probability)
    for realclass, probability in zip(real_classes_s, probabilites_s):
        print(realclass, end="")
        print(probability) 
    """
    #Gryffindor_Model.predict_proba()
    #print(Gryffindor_Model.features_coeffs)
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