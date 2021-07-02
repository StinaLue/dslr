#!/usr/bin/env python3
from ft_MachineLearning.LogisticRegression import LogisticRegression_Model
import pandas as pd
import os
from argparse import ArgumentParser
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f",
        metavar=("filename", "weights"),
        nargs=2,
        help="Filename of the dataset CSV file and of the weights CSV file")
    group.add_argument("-me",
        action="store_true",
        help="Ask for prediction of house for your own data")
    parser.add_argument("-v",
        action="store_true",
        help="Show the distribution of students this year")
    return parser.parse_args()

def other_calculate_grade(estimate, min, max):
    range_grade = abs(max - min)
    estimate = estimate / 10
    scaled_grade = range_grade * estimate
    result = scaled_grade + min
    return (result)

def get_infos():
    features_selected = ["First Name", "Astronomy", "Herbology", "Charms", "Ancient Runes"]
    column_names = features_selected
    column_values = []
    csv = []
    name = ""
    astro = -1
    herbo = -1
    charms = -1
    runes = -1
    name = input("Please enter your name : ")
    column_values.append(name)
    while not int(astro) in range(0,11):
        astro = input("Please enter your Astronomy grade on a range from 0 to 10 : ")
    astro = other_calculate_grade(int(astro), -966.74055, 1016.21194)
    column_values.append(astro)
    while not int(herbo) in range(0,11):
        herbo = input("Please enter your Herbology grade on a range from 0 to 10 : ")
    herbo = other_calculate_grade(int(herbo), -10.29566, 11.61290)
    column_values.append(herbo)
    while not int(charms) in range(0,11):
        charms = input("Please enter your Charms grade on a range from 0 to 10 : ")
    charms = other_calculate_grade(int(charms), -261.04892, -225.42814)
    column_values.append(charms)
    while not int(runes) in range(0,11):
        runes = input("Please enter your Ancient Runes grade on a range from 0 to 10 : ")
    runes = other_calculate_grade(int(runes), 283.86961, 745.39622)
    column_values.append(runes)
    csv.append(column_values)
    csv_df = pd.DataFrame(csv, columns = features_selected)
    csv_df.to_csv("me.csv", index=False)
    return("me.csv")

def create_house_model(filename, modelname, scaled_features, features_selected):
    housedata = pd.read_csv(filename)
    intercept = (housedata["Intercept"]).astype(float).to_numpy()
    features_coeffs = np.zeros((len(features_selected), 1))
    for i, column in enumerate(housedata.columns[1:]):
        features_coeffs[i] = housedata[column]
    modelname = LogisticRegression_Model(housedata, features_selected)
    probabilites = modelname.predict_proba(scaled_features, features_coeffs, intercept)
    return(probabilites)

def get_all_probas(weightsfile, scaled_features, features_selected):
    df = pd.read_csv(weightsfile)
    intercept_g = np.array([df["Intercept"][0]], dtype=float)
    intercept_h = np.array([df["Intercept"][2]], dtype=float)
    intercept_r = np.array([df["Intercept"][4]], dtype=float)
    intercept_s = np.array([df["Intercept"][6]], dtype=float)
    features_coeffs_g = np.zeros((len(features_selected), 1))
    features_coeffs_h = np.zeros((len(features_selected), 1))
    features_coeffs_r = np.zeros((len(features_selected), 1))
    features_coeffs_s = np.zeros((len(features_selected), 1))
    for i, column in enumerate(df.columns[1:]):
        features_coeffs_g[i] = df[column][0]
        features_coeffs_h[i] = df[column][2]
        features_coeffs_r[i] = df[column][4]
        features_coeffs_s[i] = df[column][6]
    model = LogisticRegression_Model(df, features_selected)
    probabilities_g = model.predict_proba(scaled_features, features_coeffs_g, intercept_g)
    probabilities_h = model.predict_proba(scaled_features, features_coeffs_h, intercept_h)
    probabilities_r = model.predict_proba(scaled_features, features_coeffs_r, intercept_r)
    probabilities_s = model.predict_proba(scaled_features, features_coeffs_s, intercept_s)
    return probabilities_g, probabilities_h, probabilities_r, probabilities_s

def sortinghat(testdata, args):
    features_selected = ["Astronomy", "Herbology", "Charms", "Ancient Runes"]
    if args.me:
        scaled_test_data = (testdata - pd.Series({"Astronomy":48.155326, "Herbology":1.385517, "Charms":-243.181109, "Ancient Runes":495.937543})) / pd.Series({"Astronomy":504.119026, "Herbology":4.985124, "Charms":8.727190, "Ancient Runes":100.633136})
    else:
        scaled_test_data = (testdata - testdata.mean()) / testdata.std()
    scaled_features = scaled_test_data[features_selected].to_numpy()
    firstnames = testdata["First Name"]
    if not os.path.exists("gryffindor_weights.csv"):
        print("You have to train your model before predicting")
        exit()
    """
    probabilities_g = create_house_model("gryffindor_weights.csv", "Gryffindor", scaled_features, features_selected)
    probabilities_s = create_house_model("slytherin_weights.csv", "Slytherin", scaled_features, features_selected)
    probabilities_h = create_house_model("hufflepuff_weights.csv", "Hufflepuff", scaled_features, features_selected)
    probabilities_r = create_house_model("ravenclaw_weights.csv", "Ravenclaw", scaled_features, features_selected)
    """
    if args.me:
        probabilities_g, probabilities_h, probabilities_r, probabilities_s = get_all_probas("weights.csv", scaled_features, features_selected) 
    else:
        probabilities_g, probabilities_h, probabilities_r, probabilities_s = get_all_probas(args.f[1], scaled_features, features_selected) 

    rows = []
    for firstname, proba_G, proba_S, proba_H, proba_R in zip(firstnames, probabilities_g, probabilities_s, probabilities_h, probabilities_r):
        if args.me:
            print("Hmmmm, " + str(firstname) + "... interesting... ", end="")
        proba_tab = [proba_G, proba_S, proba_H, proba_R]
        max_proba = max(proba_tab)
        if (max_proba == proba_G):
            if args.me:
                print("GRYFFINDOR")
            else:
                rows.append("Gryffindor")
        elif (max_proba == proba_S):
            if args.me:
                print("SLYTHERIN")
            else:
                rows.append("Slytherin")
        elif (max_proba == proba_H):
            if args.me:
                print("HUFFLEPUFF")
            else:
                rows.append("Hufflepuff")
        elif (max_proba == proba_R):
            if args.me:
                print("RAVENCLAW")
            else:
                rows.append("Ravenclaw")
        else:
            if args.me:
                print(max_proba)
                print("I can't assign you to a house, you are too special.")
            else:
                rows.append("Error")
    df = pd.DataFrame(rows, columns=["Hogwarts House"])
    df.to_csv("houses.csv", index_label="Index")

    if args.v and not args.me:
        color_dict = dict({'Hufflepuff':'gold',
                  'Slytherin':'lime',
                  'Ravenclaw': 'blue',
                  'Gryffindor': 'red'})
        sns.countplot(x="Hogwarts House", palette=color_dict, data=df).set_title("Distribution of students in 2021")
        plt.show()

def main():
    args = parse_arguments()
    if args.me:
        me_filename = get_infos()
        df = read_csv(me_filename)
    else:
        df = read_csv(args.f[0])
    df = preprocess_dataframe(df)
    sortinghat(df, args)

main()
