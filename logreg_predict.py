#!/usr/bin/env python3
from ft_MachineLearning.LogisticRegression import LogisticRegression_Model
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np

#from logreg_train import read_csv

def read_csv(filename):
    verify_csv(filename)
    dataframe = pd.read_csv(filename)
    return dataframe

def preprocess_dataframe(df):
    df = df.fillna(df.mean())
    return df

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

def parse_arguments():  
    parser = ArgumentParser()
    parser.add_argument("-f",
        dest="filename",
        help="Filename of the dataset CSV file",
        required=True)
    return parser.parse_args()

def log_odds(features, coefficients, intercept):
        """
        features: numpy array holding all features used
        coefficients: numpy array holding the coefficients of each respective feature
        intercept: numpy array only holding the single value of the intercept

        log-odds are used to know the probability of a data sample belonging to the positive class
        They replace our prediction in a linear regression, and it works the same way:
        multiply each feature by its respective coefficient (weight/theta), and add the intercept

        Returns the log-odds in a numpy array
        """
        print (np.dot(features, coefficients))
        return np.dot(features, coefficients) + intercept

def sigmoid(z):
        """
        z: log-odds in a numpy array

        Maps the log-odds z into the sigmoid function.
        The sigmoid function looks like this:
        h(z) = 1 / 1 + e^(-z)
        This effectively scales our log-odds in the range [0,1].
        The results are used later to categorize easily (probability from 0 to 100% for a predicted class).

        returns sigmoid mapped log-odds in a numpy array
        """
        return 1.0 / (1.0 + np.exp(-z))

def predict_probability(features, coefficients, intercept):
        calculated_log_odds = log_odds(features, coefficients, intercept)
        probabilities = sigmoid(calculated_log_odds)
        return probabilities
        

def sortinghat(testdata):
    features_selected = ["Astronomy", "Herbology", "Charms", "Ancient Runes"]


    scaled_test_data = (testdata - testdata.mean()) / testdata.std()
    scaled_features = scaled_test_data[features_selected].to_numpy()
    firstnames = testdata["First Name"]

    gryffindordata = pd.read_csv("gryffindor_weights.csv")
    g_intercept = (gryffindordata["Intercept"]).astype(float).to_numpy()
    g_features_coeffs = np.zeros((len(features_selected), 1))
    g_features_coeffs[0] = gryffindordata["Astronomy"].astype(float).to_numpy()
    g_features_coeffs[1] = gryffindordata["Herbology"].astype(float).to_numpy()
    g_features_coeffs[2] = gryffindordata["Charms"].astype(float).to_numpy()
    g_features_coeffs[3] = gryffindordata["Ancient Runes"].astype(float).to_numpy()
    Gryffindor_Model = LogisticRegression_Model(gryffindordata, features_selected)
    probabilites_g = Gryffindor_Model.predict_proba(scaled_features, g_features_coeffs, g_intercept)

    slytherindata = pd.read_csv("slytherin_weights.csv")
    g_intercept = (slytherindata["Intercept"]).astype(float).to_numpy()
    g_features_coeffs = np.zeros((len(features_selected), 1))
    g_features_coeffs[0] = slytherindata["Astronomy"].astype(float).to_numpy()
    g_features_coeffs[1] = slytherindata["Herbology"].astype(float).to_numpy()
    g_features_coeffs[2] = slytherindata["Charms"].astype(float).to_numpy()
    g_features_coeffs[3] = slytherindata["Ancient Runes"].astype(float).to_numpy()
    Slytherin_Model = LogisticRegression_Model(slytherindata, features_selected)
    probabilites_s = Slytherin_Model.predict_proba(scaled_features, g_features_coeffs, g_intercept)
    
    hufflepuffdata = pd.read_csv("hufflepuff_weights.csv")
    g_intercept = (hufflepuffdata["Intercept"]).astype(float).to_numpy()
    g_features_coeffs = np.zeros((len(features_selected), 1))
    g_features_coeffs[0] = hufflepuffdata["Astronomy"].astype(float).to_numpy()
    g_features_coeffs[1] = hufflepuffdata["Herbology"].astype(float).to_numpy()
    g_features_coeffs[2] = hufflepuffdata["Charms"].astype(float).to_numpy()
    g_features_coeffs[3] = hufflepuffdata["Ancient Runes"].astype(float).to_numpy()
    Hufflepuff_Model = LogisticRegression_Model(hufflepuffdata, features_selected)
    probabilites_h = Hufflepuff_Model.predict_proba(scaled_features, g_features_coeffs, g_intercept)

    ravenclawdata = pd.read_csv("ravenclaw_weights.csv")
    g_intercept = (ravenclawdata["Intercept"]).astype(float).to_numpy()
    g_features_coeffs = np.zeros((len(features_selected), 1))
    g_features_coeffs[0] = ravenclawdata["Astronomy"].astype(float).to_numpy()
    g_features_coeffs[1] = ravenclawdata["Herbology"].astype(float).to_numpy()
    g_features_coeffs[2] = ravenclawdata["Charms"].astype(float).to_numpy()
    g_features_coeffs[3] = ravenclawdata["Ancient Runes"].astype(float).to_numpy()
    Ravenclaw_Model = LogisticRegression_Model(ravenclawdata, features_selected)
    probabilites_r = Ravenclaw_Model.predict_proba(scaled_features, g_features_coeffs, g_intercept)

    for firstname, proba_G, proba_S, proba_H, proba_R in zip(firstnames, probabilites_g, probabilites_s, probabilites_h, probabilites_r):
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
    
    scaled_test_data = (testdata - testdata.mean()) / testdata.std()
    scaled_features = scaled_test_data[features_selected].to_numpy()
    class_real = testdata["Hogwarts House"]
    firstnames = testdata["First Name"]
    #real_classes = (testdata["Hogwarts House"] == "Gryffindor").astype(int).to_numpy()
    probabilites_g = Gryffindor_Model.predict_proba(scaled_features, gryffindordata[features_selected], gryffindordata["Intercept"])
    #real_classes_s = (testdata["Hogwarts House"] == "Slytherin").astype(int).to_numpy()
    probabilites_s = Slytherin_Model.predict_proba(scaled_features, Slytherin_Model.features_coeffs, Slytherin_Model.intercept)

    probabilites_h = Hufflepuff_Model.predict_proba(scaled_features, Hufflepuff_Model.features_coeffs, Hufflepuff_Model.intercept)

    probabilites_r = Ravenclaw_Model.predict_proba(scaled_features, Ravenclaw_Model.features_coeffs, Ravenclaw_Model.intercept)
    
    for firstname, proba_G, proba_S, proba_H, proba_R in zip(firstnames, probabilites_g, probabilites_s, probabilites_h, probabilites_r):
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

def main():
    args = parse_arguments()
    test_df = read_csv(args.filename)
    test_df = preprocess_dataframe(test_df)
    sortinghat(test_df)

main()