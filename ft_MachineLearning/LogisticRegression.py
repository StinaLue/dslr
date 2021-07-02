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
    2. mutiply each feature coeff by its feature and add intercept to get log-odds (probability)
    3. scale the log-odds by mapping them to the our logistic function --> the sigmoid function, giving us a probability in the range (0,1)
    4. calculate the partial derivative of the log-loss of our current logistic function
    5. use the partial derivative to update our intercept and our features coefficients (gradient descent)
    6. run the gradient descent until we reach the best minimum (change the learning rate if needed)
    """
    def __init__(self, dataframe, features_selected, class_title=None, class_to_test=None):
        """
        dataframe: df to use to train or predict
        features_selected: list of features used to train or predict (shape: ["feature1", "feature2"])
        class_title: title for the class to test --> only for training
        class_to_test: specific class that we want to predict (one vs all style) --> only for training
        """
        self.dataframe = dataframe
        self.scaled_dataframe = (self.dataframe - self.dataframe.mean()) / self.dataframe.std()
        self.nb_entries = float(len(self.dataframe.index))
        self.features_selected = features_selected
        self.features_coeffs = np.zeros((len(features_selected), 1))
        self.intercept = np.zeros(1)
        if class_title is not None:
            #get labels in array as 0s and 1s for each entry of our class if the entry == class_to_test
            self.real_labels = (dataframe[class_title] == class_to_test).astype(int).to_numpy()
            np.reshape(self.real_labels, (int(self.nb_entries), 1))
        else:
            self.real_labels = None
        self.features = {}
        for i in range(len(self.features_selected)):
            self.features[self.features_selected[i]] = self.dataframe[self.features_selected[i]].to_numpy()
        self.scaled_features = {}
        for i in range(len(self.features_selected)):
            self.scaled_features[self.features_selected[i]] = self.scaled_dataframe[self.features_selected[i]].to_numpy()

        self.features_matrix = dataframe[features_selected].to_numpy()
        self.scaled_features_matrix = self.scaled_dataframe[features_selected].to_numpy()

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
        Calculate the cost of our current sigmoid function through the log-loss formula where:
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

    def train(self, features_matrix, labels_array, learning_rate, batch_size, iterations, tolerance):
        """
        runs the Logistic regression algorithm with all the provided class attributes
        """
        for i in range(iterations): #stop range if update is small enough (tolerance), or range limit is reached (iterations)
            current_batchlabels, current_batchfeatures = self.selrand_features_labels(features_matrix, labels_array, batch_size)
            current_predictproba = self.predict_proba(current_batchfeatures, self.features_coeffs, self.intercept)
            #partial deriv wrt intercept (tmpθ0) = ratioDApprentissage ∗ 1/m (m−1∑i=0)(predicted_proba(i) - real_class(i))
            intercept_tmp = learning_rate * (1/batch_size) * np.sum(current_predictproba - current_batchlabels)
            features_gradient_tmp = []
            #modify the shape of the features np array to work with np dot
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
