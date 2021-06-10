#!/usr/bin/env python3

from ft_MachineLearning.LogisticRegression import LogisticRegression_Model
import pandas as pd

def preprocess_dataframe(df):
    df = df.fillna(df.mean())
    return df

def sortinghat():
    features_selected = ["Astronomy", "Herbology", "Charms", "Ancient Runes"]
    testdata = pd.read_csv("dataset_test.csv")

    gryffindordata = pd.read_csv("gryffindor_weights.csv")
    Gryffindor_Model = LogisticRegression_Model(gryffindordata, "Gryffindor", "Hogwarts House", features_selected)
    
    slytherindata = pd.read_csv("slytherin_weights.csv")
    Slytherin_Model = LogisticRegression_Model(slytherindata, "Slytherin", "Hogwarts House", features_selected)
    
    ravenclawdata = pd.read_csv("ravenclaw_weights.csv")
    Ravenclaw_Model = LogisticRegression_Model(ravenclawdata, "Ravenclaw", "Hogwarts House", features_selected)
    
    hufflepuffdata = pd.read_csv("hufflepuff_weights.csv")
    Hufflepuff_Model = LogisticRegression_Model(hufflepuffdata, "Hufflepuff", "Hogwarts House", features_selected)
    
    testdata = preprocess_dataframe(testdata)
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

def main():
    args = parse_arguments()
    #theta0, theta1 = get_thetas()
    #num = input("Please enter a number: ")
    #check_is_digit(num)
    #predict_cost(num, theta0, theta1)

main()