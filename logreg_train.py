#!/usr/bin/env python3

from ft_MachineLearning.LogisticRegression import LogisticRegression_Model
from argparse import ArgumentParser
import os

def parse_arguments():  
    parser = ArgumentParser()
    parser.add_argument("-f",
        dest="filename",
        help="add csv flag",
        required=True)
    parser.add_argument("-v", "--verbose",
        action="store_true",
        dest="verbose",
        default=False,
        help="Give a verbose output of the linear regression training")
    parser.add_argument("--features",
        dest="features",
        default=None,
        help="x and y features to use in the given file, shape is 'xfeaturename,yfeaturename'")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.features is None:
        data = LinearRegression_Model(args.filename, verbose=args.verbose)
    else:
        try:
            xfeature = args.features.split(',')[0]
            yfeature = args.features.split(',')[1]
            data = LinearRegression_Model(args.filename, xfeature=xfeature, yfeature=yfeature, verbose=args.verbose)
        except:
            print("Something went wrong with the features you gave.")
            exit()

main()