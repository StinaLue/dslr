# CodeCademy : def log_loss(probabilities,actual_class):
# return np.sum(-(1/actual_class.shape[0])*(actual_class*np.log(probabilities) + (1-actual_class)*np.log(1-probabilities)))

# return (np.where(probabilities >= threshold, 1, 0))
"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):
    predictions = sigmoid(X @ theta)
    predictions[predictions == 1] = 0.999 # log(1)=0 causes error in division
    error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions);
    return sum(error) / len(y);

def cost_gradient(theta, X, y):
    predictions = sigmoid(X @ theta);
    return X.transpose() @ (predictions - y) / len(y)


X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
X[:, 1:] = x

classifiers = np.zeros(shape=(numLabels, numFeatures + 1))

for c in range(0, numLabels):
    label = (y == c).astype(int)
    initial_theta = np.zeros(X.shape[1])
    classifiers[c, :] = opt.fmin_cg(cost, initial_theta, cost_gradient, (X, label), disp=0)




predictions = classProbabilities.argmax(axis=1)

print("Training accuracy:", str(100 * np.mean(predictions == y)) + "%")
    """