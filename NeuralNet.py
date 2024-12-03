import csv
import random
from PreprocessData import PreprocessData


class NeuralNet:
    def __init__(self, L, n, epochs=None, learning_rate=None, momentum=None, fact=None, validation_split=None):
        """
        Task 2: Implement a neural network with back-propagation manually.
        """

        self.L = L  # Number of layers
        self.n = n.copy()  # Number of units in each layer
        self.epochs = epochs  # Number of epochs
        self.learning_rate = learning_rate  # Learning rate
        self.momentum = momentum  # Momentum term
        self.fact = fact  # Activation function
        self.validation_split = validation_split  # Percentage of validation set

        # Online BP algorithm: 2- Initialize all weights and thresholds randomly
        self.w = []
        for i in range(L - 1):
            self.w.append([[random.uniform(-1.0, 1.0) for _ in range(self.n[i + 1])] for _ in range(self.n[i])])
        self.theta = []
        for i in range(L - 1):
            self.theta.append([random.uniform(-1.0, 1.0) for _ in range(self.n[i + 1])])
        self.d_w_prev = []
        for i in range(L - 1):
            self.d_w_prev.append([[0.0 for _ in range(self.n[i + 1])] for _ in range(self.n[i])])
        self.d_theta_prev = []
        for i in range(L - 1):
            self.d_theta_prev.append([0.0 for _ in range(self.n[i + 1])])

        print("NeuralNet initialized.")

    def fit(self, X=None, y=None):
        """
        Subtask 2.1: Use training and validation data to fit neural network
        """

        # Online BP algorithm: 1- Scale input and/or output patterns, if needed
        X_train, X_val = [[0], [0]]
        y_train, y_val = [[0], [0]]

        # TODO: For epoch = 1 To num epochs
        for epoch in range(self.epochs):

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
        # TODO: Return the evolution of the training and validation errors for each epoch

    def main(self):
        """
        Main function to execute all subtasks of Neural Network with Back-Propagation (BP).
        """
        print("Executing all subtasks of Implementation of BP...")

        # Task 2: Implementation of BP
        self.fit()

def readFile(filepath='./data/transformed_train_matrix.csv'):
    train_matrix = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        # Skip the header if it exists
        next(reader, None)
        for row in reader:
            train_matrix.append([float(value) for value in row])


if __name__ == "__main__":
    preprocessData = PreprocessData()
    preprocessData.select_and_analyze_dataset()

    nn = NeuralNet(4, [4, 9, 5, 1], 100)
    nn.main()
