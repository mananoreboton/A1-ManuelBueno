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

        # Online BP algorithm: L2 - Initialize all weights and thresholds randomly
        self.w = [None]
        self.theta = [None]
        for l in range(1, self.L):
            self.w.append(np.random.randn(self.n[l], self.n[l - 1]))
            self.theta.append(np.random.randn(self.n[l], 1))
        self.d_w_prev = [0.0]
        self.d_theta_prev = [0.0]
        for l in range(1, self.L):
            self.d_w_prev.append(np.zeros((self.n[l], self.n[l - 1])))
            self.d_theta_prev.append(np.zeros((self.n[l], 1)))

        self.h = [0.0] * self.L
        self.xi = [0.0] * self.L
        self.delta = [0.0] * self.L
        self.d_w = [0.0] * self.L
        self.d_theta = [0.0] * self.L
        self.activation_function, self.activation_derivative = self.get_activation_function(self.fact)
        self.training_errors = []
        self.validation_errors = []

        print(f"NeuralNet initialized with self.L = '{self.L}', self.n = '{self.n}', self.n_epochs = '{self.n_epochs}', self.learning_rate = '{self.learning_rate}', self.momentum = '{self.momentum}', self.fact = '{self.fact}', self.validation_split = '{self.validation_split}'")

    def fit(self, X, y):
        """
        Subtask 2.1: Use training and validation data to fit neural network
        """
        num_samples = X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        split_idx = int(num_samples * (1 - self.validation_split))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        num_train_samples = X_train.shape[0]

        # Online BP algorithm: L3 - For epoch = 1 To num epochs
        for epoch in range(self.n_epochs):
            indices = np.arange(num_train_samples)
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            # Online BP algorithm: L4 - For pat = 1 To num training patterns
            for pat in range(num_train_samples):

                # Online BP algorithm: L5 - Choose a random pattern (xμ, zμ) of the training set
                idx = np.random.randint(num_train_samples)
                x_mu = X_train_shuffled[idx].reshape(-1, 1)
                z_mu = y_train_shuffled[idx].reshape(-1, 1)
                # Online BP algorithm: L6 - Feed−forward propagation of pattern xμ to obtain the output o(xμ)
                self.feedforward(x_mu)
                # Online BP algorithm: L7 - Back−propagate the error for this pattern
                self.backpropagate(z_mu)
                # Online BP algorithm: L8 - Update the weights and thresholds
                self.update_weights()

            # Online BP algorithm: L10 - Feed−forward all training patterns and calculate their prediction quadratic error
            train_error = self.calculate_error(X_train, y_train)
            # Online BP algorithm: L11 - Feed−forward all validation patterns and calculate their prediction quadratic error
            val_error = self.calculate_error(X_val, y_val)
            self.training_errors.append(train_error)
            self.validation_errors.append(val_error)

            # print(f'Epoc {epoch + 1}/{self.n_epochs}, Train Error: {train_error}, Validation Error: {val_error}')

    def predict(self, X):
        """
        Subtask 2.2: Use test data to predict labels
        """
        # Online BP algorithm: L14 - Feed−forward all test pattern
        num_samples = X.shape[0]
        outputs = []
        for i in range(num_samples):
            x = X[i].reshape(-1, 1)
            self.feedforward(x)
            output = self.xi[-1].flatten()
            outputs.append(output)
        return np.array(outputs)

    def oss_epochs(self):
        """
        Subtask 2.3: Return the evolution of the training and validation errors for each epoch
        """
        return np.array(self.training_errors), np.array(self.validation_errors)

    def calculate_error(self, X, y):
        num_samples = X.shape[0]
        total_error = 0
        for i in range(num_samples):
            x = X[i].reshape(-1, 1)
            target = y[i].reshape(-1, 1)
            self.feedforward(x)
            output = self.xi[-1]
            total_error += np.sum((output - target) ** 2)
        return total_error / num_samples

    def feedforward(self, x):
        self.xi[0] = x
        for l in range(1, self.L):
            self.h[l] = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            self.xi[l] = self.activation_function(self.h[l])

    def backpropagate(self, z):
        l = self.L - 1
        self.delta[l] = (self.xi[l] - z) * self.activation_derivative(self.h[l])
        for l in range(self.L - 2, 0, -1):
            self.delta[l] = np.dot(self.w[l + 1].T, self.delta[l + 1]) * self.activation_derivative(self.h[l])
        for l in range(1, self.L):
            self.d_w[l] = -self.learning_rate * np.dot(self.delta[l], self.xi[l - 1].T) + self.momentum * self.d_w_prev[l]
            self.d_theta[l] = self.learning_rate * self.delta[l] + self.momentum * self.d_theta_prev[l]

    def update_weights(self):
        for l in range(1, self.L):
            self.w[l] += self.d_w[l]
            self.theta[l] += self.d_theta[l]
            self.d_w_prev[l] = self.d_w[l]
            self.d_theta_prev[l] = self.d_theta[l]

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

    def main(self, X_train, y_train, X_prediction, y_prediction):
        """
        Main function to execute all subtasks of Neural Network with Back-Propagation (BP).
        """
        print("Executing all subtasks of Implementation of BP...")

        # Task 2: Implementation of BP
        self.fit(X_train, y_train)

        predictions = self.predict(X_prediction)
        for pred, y_pred in zip(predictions, y_prediction):
            print(f"Prediction: {pred}, Expected Prediction: {y_pred}")

        training_errors, validation_errors = self.oss_epochs()
        for train_err, val_err in zip(training_errors, validation_errors):
            print(f"Training Error: {train_err}, Validation Error: {val_err}")

if __name__ == "__main__":
    preprocessData = PreprocessData()
    preprocessData.select_and_analyze_dataset()
    X_in, y_in = preprocessData.read_transformed_data_from_file()
    X_in_prediction, y_in_prediction = preprocessData.read_transformed_data_from_file('./data/transformed_test_matrix.csv')

    neural_net = NeuralNet(
        L=3,
        n= [np.shape(X_in)[1], 10, 1],
        n_epochs=100,
        learning_rate=0.01,
        momentum=0.9,
        activation_function='tanh',
        validation_split=0.2
    )
    neural_net.main(X_in, y_in, X_in_prediction, y_in_prediction)
