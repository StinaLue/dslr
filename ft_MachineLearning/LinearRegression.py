import matplotlib.pyplot as plt
import pandas
import os

class LinearRegression_Model:
    """
    This class' goal is to run a linear regression algorithm training on a user-specified csv file.
    After the training, the linear function parameters are stored in the class as theta0 and theta1.
    """
    def __init__(self, filename, xfeature=None, yfeature=None, verbose=False):
        """
        filename: CSV file containing the data
        xfeature: name of the feature for the x axis
        yfeature: name of the feature for the y axis
        verbose: boolean --> plot during training or not
        """
        self.filename = filename
        self.dataframe = self.__read_csv(self.filename)
        self.scaled_dataframe = (self.dataframe - self.dataframe.mean()) / self.dataframe.std()
        self.nb_entries = float(len(self.dataframe.index))
        self.__set_features(xfeature, yfeature, self.dataframe)
        self.true_X, self.true_Y = self.__init_XY(self.dataframe, self.features)
        self.verbose = verbose
        self.scaled_X, self.scaled_Y = self.__init_XY(self.scaled_dataframe, self.features)
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.learning_rate = 0.1
        self.__train()
        self.__unscale_predictY()
        self.__unscale_thetas()
        self.__write_thetas()
        self.__plot_result()

    def __unscale_predictY(self):
        """
        Reverses the Z-scaling for our scaled predicted y tab
        """
        self.predictY = (self.current_scaled_predictY * self.dataframe.std()[self.features[1]]) + self.dataframe.mean()[self.features[1]]

    def __unscale_thetas(self):
        """
        Unscale theta parameters of the linear functions thx to maths --> https://www.mathsisfun.com/algebra/line-equation-2points.html
        """
        slope = (self.predictY[0] - self.predictY[1]) / (self.true_X[0] - self.true_X[1])
        y_intercept = (slope * 0) - (slope * self.true_X[0]) + self.predictY[0]
        self.theta1 = slope
        self.theta0 = y_intercept

    def __write_thetas(self):
        f = open("thetas.txt", "w")
        f.write(str(self.theta0) + " " + str(self.theta1))
        f.close()
        print("t0 " + str(self.theta0))
        print("t1 " + str(self.theta1))

    def __plot_result(self):
        """
        plots the end result of our training.
        """
        try:
            plt.scatter(self.true_X, self.true_Y)
            plt.plot(self.true_X, self.predictY, color='red')  # regression line
            plt.xlabel(self.features[0])
            plt.ylabel(self.features[1])
            plt.title(self.filename)
            plt.show()
        except KeyboardInterrupt:
            print("Ctrl-c received. Goodbye !")
            exit()

    def __train(self):
        """
        runs the linear regression algorithm with all the provided class attributes
        """
        current_error_tmp = 0
        counter = 0
        for i in range(1000): #stop range if mean_squared_error is 3 x the same etc.
            self.current_scaled_predictY = self.__hypothesis_function(self.scaled_X)
            #tmpθ0 = ratioDApprentissage ∗ 1/m (m−1∑i=0)(prixEstime(kilométrage[i]) − prix[i])
            theta0_tmp = self.learning_rate * (1/self.nb_entries) * sum(self.current_scaled_predictY - self.scaled_Y)
            #tmpθ1= ratioDApprentissage ∗ 1/m (m−1∑i=0)(prixEstime(kilométrage[i]) − prix[i]) ∗ kilométrage[i]
            theta1_tmp = self.learning_rate * (1/self.nb_entries) * sum((self.current_scaled_predictY - self.scaled_Y) * self.scaled_X)
            self.theta0 = self.theta0 - theta0_tmp
            self.theta1 = self.theta1 - theta1_tmp
            current_error = self.__mean_squared_error(self.scaled_Y, self.__hypothesis_function(self.scaled_X))
            if current_error == current_error_tmp:
                counter += 1
            else:
                counter = 0
            if counter > 10:
                print("Minimum cost found, stopped training")
                break
            current_error_tmp = current_error
            if self.verbose is True:
                try:
                    #offset = 0.5
                    offset = 0.1
                    plt.cla()
                    plt.scatter(self.scaled_X, self.scaled_Y)
                    plt.plot(self.scaled_X, self.current_scaled_predictY, color='red', marker='o')  # regression line
                    for j in range(len(self.scaled_X)):
                        plt.plot([self.scaled_X[j], self.scaled_X[j]], [self.scaled_Y[j], self.current_scaled_predictY[j]], color='yellow')
                    plt.axis([min(self.scaled_X) - offset, max(self.scaled_X) + offset, \
                        min(self.scaled_Y) - offset, max(self.scaled_Y) + offset])
                    plt.axis("off")
                    plt.pause(0.0001)
                    print("current error : "+str(current_error))
                except KeyboardInterrupt:
                    print("The program was quitted before completion. Therefore the thetas were not saved")
                    exit()
        if self.verbose is True:
            plt.show()
            plt.close()

    def __mean_squared_error(self, true_Y, predict_Y):
        """
        Return the 'cost' of the current predictions made by the linear function
        """
        tmp = (true_Y - predict_Y) ** 2
        return tmp.mean()

    def __set_features(self, xfeature, yfeature, dataframe):
        """
        set appropriate x and y features as class attributes.
        thoses features can be user-defined during object initialization.
        """
        if xfeature is not None and yfeature is not None:
            self.features = [xfeature, yfeature]
        else:
            self.features = [dataframe.columns[0], dataframe.columns[1]]

    def __init_XY(self, dataframe, features):
        """
        test provided features in case they seem abnormal.
        """
        try:
            tmp_X = dataframe[features[0]].to_numpy()
        except:
            print("Wrong X feature given")
            exit()
        try:
            tmp_Y = dataframe[features[1]].to_numpy()
        except:
            print("Wrong Y feature given")
            exit()
        return tmp_X, tmp_Y

    def __verify_csv(self, filename):
        """
        very basic check for the CSV file --> it has to exist, and end in .csv
        """
        if not os.path.exists(filename):
            print("The file %s does not exist!" % filename)
            exit()
        if not filename.endswith('.csv'):
            print("The file %s is not a csv file!" % filename)
            exit()

    def __read_csv(self, filename):
        self.__verify_csv(filename)
        dataframe = pandas.read_csv(filename)
        return dataframe

    def __hypothesis_function(self, x):
        """
        linear function defined by the class thetas and a given x.
        """
        return self.theta0 + self.theta1 * x
    
    def predict(self, x):
        """
        prints a y prediction with the trained parameters kept in the class.
        """
        y_prediction = self.theta0 + self.theta1 * x
        print("Prediction for x = " + str(x) + " is --> y = " + str(y_prediction))
    

if __name__ == "__main__":
    from argparse import ArgumentParser
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
        return parser.parse_args()
    def main():
        args = parse_arguments()
        data = LinearRegression_Model(args.filename, verbose=args.verbose)

    main()