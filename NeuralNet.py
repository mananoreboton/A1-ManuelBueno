import random
from PreprocessData import PreprocessData
import numpy as np

class NeuralNet:
    def __init__(self, L, n, n_epochs, learning_rate, momentum, activation_function, validation_split):
        self.L = L
        self.n = n
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.fact = activation_function
        self.validation_split = validation_split

        # Online BP algorithm: 2- Initialize all weights and thresholds randomly
        self.w = [None]
        self.theta = [None]
        for l in range(1, self.L):
            self.w.append(np.random.randn(self.n[l], self.n[l - 1]))
            self.theta.append(np.random.randn(self.n[l], 1))
        self.d_w_prev = [None]
        self.d_theta_prev = [None]
        for l in range(1, self.L):
            self.d_w_prev.append(np.zeros((self.n[l], self.n[l - 1])))
            self.d_theta_prev.append(np.zeros((self.n[l], 1)))

        self.h = [None] * self.L
        self.xi = [None] * self.L
        self.delta = [None] * self.L
        self.d_w = [None] * self.L
        self.d_theta = [None] * self.L
        self.activation_function, self.activation_derivative = self.get_activation_function(self.fact)
        self.training_errors = []
        self.validation_errors = []

        print("NeuralNet initialized.")

    def fit(self, X=None, y=None):
        """
        Subtask 2.1: Use training and validation data to fit neural network
        """

        # Online BP algorithm: 1- Scale input and/or output patterns, if needed
        X_train, X_val = [[0], [0]]
        y_train, y_val = [[0], [0]]

        # TODO: For epoch = 1 To num epochs
        for epoch in range(self.n_epochs):

            # TODO: For pat = 1 To num training patterns
            indices = list(range(len(X_train)))
            random.shuffle(indices)
            for i in indices:
                pass

            # TODO: Choose a random pattern (xμ, zμ) of the training set
            # TODO: Feed−forward propagation of pattern xμ to obtain the output o(xμ)
            # TODO: Back−propagate the error for this pattern
            # TODO: Update the weights and thresholds
            # TODO: Feed−forward all training patterns and calculate their prediction quadratic error
            # TODO: Feed−forward all validation patterns and calculate their prediction quadratic error
        pass

    def predict(self, X):
        """
        Subtask 2.2: Use test data to predict labels
        """
        # TODO: Feed−forward all test patterns
        pass

    def oss_epochs(self):
        """
        Subtask 2.3: Return the evolution of the training and validation errors for each epoch
        """
        return np.array(self.training_errors), np.array(self.validation_errors)

    def get_activation_function(self, name):
        if name == 'sigmoid':
            def activation(x):
                return 1 / (1 + np.exp(-x))
            def derivative(x):
                s = activation(x)
                return s * (1 - s)
        elif name == 'tanh':
            def activation(x):
                return np.tanh(x)
            def derivative(x):
                return 1 - np.tanh(x) ** 2
        elif name == 'relu':
            def activation(x):
                return np.maximum(0, x)
            def derivative(x):
                return np.where(x > 0, 1.0, 0.0)
        elif name == 'linear':
            def activation(x):
                return x
            def derivative(x):
                return np.ones_like(x)
        else:
            raise ValueError('Unsupported activation function name')
        return activation, derivative

    def main(self):
        """
        Main function to execute all subtasks of Neural Network with Back-Propagation (BP).
        """
        print("Executing all subtasks of Implementation of BP...")

        # Task 2: Implementation of BP
        self.fit()

def readFile(filepath='./data/transformed_train_matrix.csv'):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    return data

if __name__ == "__main__":
    preprocessData = PreprocessData()
    preprocessData.select_and_analyze_dataset()
    data_train = readFile()
    X_in = data_train[:, :-1]
    y_in = data_train[:, -1]

    nn = NeuralNet(
        L=3,
        n= [np.shape(X_in)[1], 10, 1],
        n_epochs=100,
        learning_rate=0.01,
        momentum=0.9,
        activation_function='tanh',
        validation_split=0.2
    )
    nn.oss_epochs()
