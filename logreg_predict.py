#!/usr/bin/env python3
from ft_MachineLearning.LogisticRegression import LogisticRegression_Model
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import accuracy_score

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

    rows = []
    for firstname, proba_G, proba_S, proba_H, proba_R in zip(firstnames, probabilites_g, probabilites_s, probabilites_h, probabilites_r):
        #print("Hmmmm, " + str(firstname) + " is interesting... ", end="")
        proba_tab = [proba_G, proba_S, proba_H, proba_R]
        max_proba = max(proba_tab)
        if (max_proba == proba_G):
            rows.append("Gryffindor")
        elif (max_proba == proba_S):
            rows.append("Slytherin")
        elif (max_proba == proba_H):
            rows.append("Hufflepuff")
        elif (max_proba == proba_R):
            rows.append("Ravenclaw")
        else:
            rows.append("Error")
    df = pd.DataFrame(rows, columns=["Hogwarts House"])
    df.to_csv("houses.csv", index_label="Index")

def main():
    args = parse_arguments()
    df = read_csv(args.filename)
    df = preprocess_dataframe(df)
    sortinghat(df)

main()