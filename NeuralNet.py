import numpy as np

"""
Task 2: Implement a neural network with back-propagation manually.
"""
class NeuralNet:
    def __init__(self, L, n, epochs=None, learning_rate=None, momentum=None, fact=None, validation_split=None):
        """
        Initialize PreprocessData.
        """

        self.L = L  # Number of layers
        self.n = n.copy()  # Number of units in each layer
        self.epochs = epochs  # Number of epochs
        self.learning_rate = learning_rate  # Learning rate
        self.momentum = momentum  # Momentum term
        self.fact = fact  # Activation function
        self.validation_split = validation_split  # Percentage of validation set

        # Subtask 1: Initialize all weights and thresholds randomly
        self.w = [np.random.randn(self.n[i], self.n[i + 1]) for i in range(L - 1)]
        self.theta = [np.random.randn(self.n[i + 1]) for i in range(L - 1)]
        self.d_w_prev = [np.zeros((self.n[i], self.n[i + 1])) for i in range(L - 1)]
        self.d_theta_prev = [np.zeros(self.n[i + 1]) for i in range(L - 1)]

        print("NeuralNet initialized.")

    def fit(self, X=None, y=None):
        """
        Subtask 2.1: Use training and validation data to fit neural network
        """
        # TODO: Scale input and/or output patterns, if needed
        # TODO: To initialize all weights and thresholds randomly
        # TODO: For epoch = 1 To num epochs
        # TODO: For pat = 1 To num training patterns
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
        # TODO: Return the evolution of the training and validation errors for each epoch

    def main(self):
        """
        Main function to execute all subtasks of Neural Network with Back-Propagation (BP).
        """
        print("Executing all subtasks of Implementation of BP...")

        # Task 2: Implementation of BP
        self.fit()


if __name__ == "__main__":

    nn = NeuralNet(4, [4, 9, 5, 1])
    nn.main()
