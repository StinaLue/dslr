import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

class LogisticRegression_Model:
    """
    This class' goal is to run a Logistic regression algorithm training on a user-specified csv file.
    After the training, the Logistic function parameters are stored in the class as theta0 and theta1.

    How it works:
    1. init all features coeffs and intercept to 0 
    2. mutiply each feature coeff by its feature to get log-odds (probability)
    3. scale the log-odds by mapping them to the our logistic function --> the sigmoid function, giving us a probability in the range (0,1)
    4. calculate the partial derivative of the log-loss of our current logistic function
    5. use the partial derivative to update our intercept and our features coefficients (gradient descent)
    6. run the gradient descent until we reach the best minimum (change the learning rate if needed)
    """
    def __init__(self, dataframe, features_selected, class_name=None, class_to_test=None):
        """
        filename: CSV file containing the data
        xfeature: name of the feature for the x axis
        yfeature: name of the feature for the y axis
        verbose: boolean --> plot during training or not
        """
        self.dataframe = dataframe
        self.scaled_dataframe = (self.dataframe - self.dataframe.mean()) / self.dataframe.std()
        self.nb_entries = float(len(self.dataframe.index))
        self.features_selected = features_selected
        self.features_coeffs = np.zeros((len(features_selected), 1))
        self.intercept = np.zeros(1)
        self.scaled_intercept = np.zeros(1)
        #INT OR FLOAT FOR 0 and 1 ??
        if class_name is not None:
            self.classes = (dataframe[class_name] == class_to_test).astype(int).to_numpy()
        else:
            self.classes = None
        np.reshape(self.classes, (int(self.nb_entries), 1))
        self.features = {}
        for i in range(len(self.features_selected)):
            self.features[self.features_selected[i]] = self.dataframe[self.features_selected[i]].to_numpy()
        self.scaled_features = {}
        for i in range(len(self.features_selected)):
            self.scaled_features[self.features_selected[i]] = self.scaled_dataframe[self.features_selected[i]].to_numpy()

        self.features_matrix = dataframe[features_selected].to_numpy()
        self.scaled_features_matrix = self.scaled_dataframe[features_selected].to_numpy()

        self.dict_feat_label = {"Features":self.scaled_features_matrix, "Labels":self.classes}

    def select_random_matrix_batch(self, features_matrix, batch_size):
        """
        features_matrix: numpy 2D array
        batch_size: number of random rows in the new numpy 2D array

        selects batch_size of random rows inside the 2D numpy array features_matrix
        returns the 2D numpy array batch
        Very useful for Stochastic (for batch size == 1) or Batch Gradient Descent
        """
        new_random_batch = features_matrix[np.random.choice(features_matrix.shape[0], batch_size, replace=False)]
        return new_random_batch

    def selrand_features_labels(self, features_matrix, labels_array, batch_size):
        """
        features_matrix: numpy 2D array holding features (1 row = 1 data entry with several features)
        labels = numpy array holding labels
        batch_size: number of random rows in the new numpy 2D array

        selects batch_size of random rows inside the 2D numpy array features_matrix and labels array
        returns the labels batch, and the 2D features batch
        Very useful for Stochastic (for batch size == 1) or Batch Gradient Descent
        """
        assert len(features_matrix) == len(labels_array)
        shuffler = np.random.permutation(len(features_matrix))
        batch_shuffler = shuffler[:batch_size]
        labels_batch = labels_array[batch_shuffler].reshape(batch_size, 1)
        features_batch = features_matrix[batch_shuffler]
        return labels_batch, features_batch

    def log_odds(self, features, coefficients, intercept):
        """
        features: numpy array holding all features used
        coefficients: numpy array holding the coefficients of each respective feature
        intercept: numpy array only holding the single value of the intercept

        log-odds are used to know the probability of a data sample belonging to the positive class
        They replace our prediction in a linear regression, and it works the same way:
        multiply each feature by its respective coefficient (weight/theta), and add the intercept

        Returns the log-odds in a numpy array
        """
        return np.dot(features, coefficients) + intercept

    def sigmoid(self, z):
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
    
    def log_loss(self, predicted_probabilities, actual_class):
        """
        Caculate the cost of our current sigmoid function through the log-loss formula where:
        'm': the number of data samples
        'y(i)': the class of data sample i (ex: Gryffindor(1)/Not Gryffindor(0))
        'z(i)': the log-odds of sample i
        'h(z(i))': is the sigmoid of z(i), so the probability of sample i belonging to class 1 (range [0,1])

        Full formula is:
        −1/m * ​i=1|∑|m[ y(i) * log(h(z(i))) + (1 − y(i)) * log(1 − h(z(i))) ]

        When y(i) == 1, the right part of the equation dissapears
        When y(i) == 0, the left part of the equation dissapears

        The log loss function is very punitive when the prediction is very far from the correct result (being either 1, or 0)
        """
        return np.sum(-(1 / actual_class.shape[0]) * (actual_class * np.log(predicted_probabilities) + (1 - actual_class) * np.log(1 - predicted_probabilities)))

    def predict_class(self, features, coefficients, intercept, threshold):
        calculated_log_odds = self.log_odds(features,coefficients, intercept)
        probabilities = self.sigmoid(calculated_log_odds)
        return (np.where(probabilities >= threshold, 1, 0))

    def predict_proba(self, features, coefficients, intercept):
        calculated_log_odds = self.log_odds(features, coefficients, intercept)
        probabilities = self.sigmoid(calculated_log_odds)
        return probabilities

    def __logistic_function(self, features, coefficients, intercept):
        log_odds = np.dot(features, coefficients) + intercept
        results = self.sigmoid(log_odds)

    #def unscale_predictY(self):
        """
        Reverses the Z-scaling for our scaled predicted y tab
        """
    #    self.predictY = (self.current_scaled_predictY * self.dataframe.std()[self.features[1]]) + self.dataframe.mean()[self.features[1]]

    def unscale_thetas(self):
        """
        Unscale theta parameters of the Logistic functions thx to maths --> https://www.mathsisfun.com/algebra/line-equation-2points.html
        """
        self.features_coeffs = (self.predictY[0] - self.predictY[1]) / (self.true_X[0] - self.true_X[1])
        for i in self.features_coeffs:
            self.intercept = (self.features_coeffs[i] * 0) - (self.features_coeffs[i] * self.true_X[0]) + self.predictY[0]
        print (self.intercept)
        print (self.features_coeffs)

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


    #def train(self, features, true_labels, learning_rate, batch_size):
    def train(self, features_matrix, labels_array, learning_rate, batch_size, iterations, tolerance):
        """
        runs the Logistic regression algorithm with all the provided class attributes
        """
        current_error_tmp = 0
        #precision = 0.00000001
        for i in range(iterations): #stop range if log_loss is nearly the same 2 consecutive times, or range limit is reached
            #self.current_batch = self.select_random_matrix_batch(self.scaled_features_matrix, batch_size)
            current_batchlabels, current_batchfeatures = self.selrand_features_labels(features_matrix, labels_array, batch_size)
            #current_predictproba = self.predict_proba(self.current_batch, self.scaled_features_coeffs, self.intercept)
            current_predictproba = self.predict_proba(current_batchfeatures, self.features_coeffs, self.intercept)
            #partial deriv wrt intercept (tmpθ0) = ratioDApprentissage ∗ 1/m (m−1∑i=0)(predicted_proba(i) - real_class(i))
            intercept_tmp = learning_rate * (1/batch_size) * np.sum(current_predictproba - current_batchlabels)
            features_gradient_tmp = []
            transposed_features = np.transpose(current_batchfeatures)
            #partial deriv wrt featurecoeff (tmpθj) = ratioDApprentissage ∗ 1/m (m−1∑i=0)(predicted_proba(i) - real_class(i)) * feature_j_value(i)
            for j in range(current_batchfeatures.shape[1]):
                features_gradient_tmp.append(learning_rate * (1/batch_size) * np.dot(transposed_features[j], (current_predictproba - current_batchlabels)))

            if (abs(intercept_tmp) <= tolerance):
                if (np.all(np.abs(features_gradient_tmp) <= tolerance)):
                    print("Tolerance reached")
                    break

            self.intercept = self.intercept - intercept_tmp
            for j in range(current_batchfeatures.shape[1]):
                self.features_coeffs[j] = self.features_coeffs[j] - features_gradient_tmp[j]

    def save_weights_csv(self, filename):
        column_names = ["Intercept"] + self.features_selected
        features_coeffs = self.features_coeffs.reshape(1, len(self.features_selected))
        column_values = np.insert(features_coeffs, 0, self.intercept, axis=1)
        csv_df = pd.DataFrame(column_values, columns=column_names)
        csv_df.to_csv(filename, index=False)

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
            help="Give a verbose output of the Logistic regression training")
        return parser.parse_args()
    def main():
        args = parse_arguments()
        data = LogisticRegression_Model(args.filename, verbose=args.verbose)

    main()