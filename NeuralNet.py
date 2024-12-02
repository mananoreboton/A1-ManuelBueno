import numpy as np

class NeuralNet:
    def __init__(self, L, n, epochs=None, learning_rate=None, momentum=None, fact=None, validation_split=None):
        self.L = L  # Number of layers
        self.n = n.copy()  # Number of units in each layer
        self.epochs = epochs  # Number of epochs
        self.learning_rate = learning_rate  # Learning rate
        self.momentum = momentum  # Momentum term
        self.fact = fact  # Activation function
        self.validation_split = validation_split  # Percentage of validation set

        self.xi = []
        for lay in range(self.L):
            self.xi.append(np.zeros(self.n[lay]))

        self.w = []
        self.w.append(np.zeros((1, 1)))
        for lay in range(1, self.L):
            self.w.append(np.zeros((self.n[lay], self.n[lay - 1])))

if __name__ == "__main__":

    nn = NeuralNet(4, [4, 9, 5, 1])
    
    print("L = ", nn.L, end="\n")
    print("n = ", nn.n, end="\n")
    
    print("xi = ", nn.xi, end="\n")
    print("xi[0] = ", nn.xi[0], end="\n")
    print("xi[1] = ", nn.xi[1], end="\n")
    
    print("wh = ", nn.w, end="\n")
    print("wh[1] = ", nn.w[1], end="\n")
