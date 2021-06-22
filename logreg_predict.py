#!/usr/bin/env python3
from ft_MachineLearning.LogisticRegression import LogisticRegression_Model
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import accuracy_score

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
        
def create_house_model(filename, modelname, scaled_features, features_selected):
    housedata = pd.read_csv(filename)
    intercept = (housedata["Intercept"]).astype(float).to_numpy()
    features_coeffs = np.zeros((len(features_selected), 1))
    for i, column in enumerate(housedata.columns[1:]):
        features_coeffs[i] = housedata[column]
    modelname = LogisticRegression_Model(housedata, features_selected)
    probabilites = modelname.predict_proba(scaled_features, features_coeffs, intercept)
    return(probabilites)

def sortinghat(testdata):
    features_selected = ["Astronomy", "Herbology", "Charms", "Ancient Runes"]

    scaled_test_data = (testdata - testdata.mean()) / testdata.std()
    scaled_features = scaled_test_data[features_selected].to_numpy()
    firstnames = testdata["First Name"]
    probabilites_g = create_house_model("gryffindor_weights.csv", "Gryffindor", scaled_features, features_selected)
    probabilites_s = create_house_model("slytherin_weights.csv", "Slytherin", scaled_features, features_selected)
    probabilites_h = create_house_model("hufflepuff_weights.csv", "Hufflepuff", scaled_features, features_selected)
    probabilites_r = create_house_model("ravenclaw_weights.csv", "Ravenclaw", scaled_features, features_selected)

    """
    gryffindordata = pd.read_csv("gryffindor_weights.csv")
    g_intercept = (gryffindordata["Intercept"]).astype(float).to_numpy()
    g_features_coeffs = np.zeros((len(features_selected), 1))
    for i, column in enumerate(gryffindordata.columns[1:]):
        g_features_coeffs[i] = gryffindordata[column]
    Gryffindor_Model = LogisticRegression_Model(gryffindordata, features_selected)
    probabilites_g = Gryffindor_Model.predict_proba(scaled_features, g_features_coeffs, g_intercept)
    
    slytherindata = pd.read_csv("slytherin_weights.csv")
    s_intercept = (slytherindata["Intercept"]).astype(float).to_numpy()
    s_features_coeffs = np.zeros((len(features_selected), 1))
    for i, column in enumerate(slytherindata.columns[1:]):
        s_features_coeffs[i] = slytherindata[column]
    Slytherin_Model = LogisticRegression_Model(slytherindata, features_selected)
    probabilites_s = Slytherin_Model.predict_proba(scaled_features, s_features_coeffs, s_intercept)
    
    hufflepuffdata = pd.read_csv("hufflepuff_weights.csv")
    h_intercept = (hufflepuffdata["Intercept"]).astype(float).to_numpy()
    h_features_coeffs = np.zeros((len(features_selected), 1))
    for i, column in enumerate(hufflepuffdata.columns[1:]):
        h_features_coeffs[i] = hufflepuffdata[column]
    Hufflepuff_Model = LogisticRegression_Model(hufflepuffdata, features_selected)
    probabilites_h = Hufflepuff_Model.predict_proba(scaled_features, h_features_coeffs, h_intercept)

    ravenclawdata = pd.read_csv("ravenclaw_weights.csv")
    r_intercept = (ravenclawdata["Intercept"]).astype(float).to_numpy()
    r_features_coeffs = np.zeros((len(features_selected), 1))
    for i, column in enumerate(ravenclawdata.columns[1:]):
        r_features_coeffs[i] = ravenclawdata[column]
    Ravenclaw_Model = LogisticRegression_Model(ravenclawdata, features_selected)
    probabilites_r = Ravenclaw_Model.predict_proba(scaled_features, r_features_coeffs, r_intercept)
    """

    rows = []
    for firstname, proba_G, proba_S, proba_H, proba_R in zip(firstnames, probabilites_g, probabilites_s, probabilites_h, probabilites_r):
        print("Hmmmm, " + str(firstname) + " is interesting... ", end="")
        proba_tab = [proba_G, proba_S, proba_H, proba_R]
        max_proba = max(proba_tab)
        if (max_proba == proba_G):
            rows.append("Gryffindor")
            print("Sorting Hat says : 'GRYFFINDOR !!!'")
        elif (max_proba == proba_S):
            rows.append("Slytherin")
            print("Sorting Hat says : 'SLYTHERIN !!!'")
        elif (max_proba == proba_H):
            print("Sorting Hat says : 'HUFFLEPUFF !!!'")
            rows.append("Hufflepuff")
        elif (max_proba == proba_R):
            print("Sorting Hat says : 'RAVENCLAW !!!'")
            rows.append("Ravenclaw")
        else:
            print("Sorting Hat says : 'am confoos'")
            rows.append("NOTHING?")
    df = pd.DataFrame(rows, columns=["Hogwarts House"])
    df.to_csv("houses.csv", index_label="Index")
    
    """csv_df = pd.DataFrame(max_proba, columns="House")
    csv_df.to_csv("test.csv")
    """
    """column_names = ["Index"] + ["Hogwarts House"]
    column_values = np.insert(features_coeffs, 0, self.intercept, axis=1)
    csv_df = pd.DataFrame(column_values, columns=column_names)
    csv_df.to_csv(filename, index=False)
    """

def main():
    args = parse_arguments()
    df = read_csv(args.filename)
    df = preprocess_dataframe(df)
    sortinghat(df)

main()