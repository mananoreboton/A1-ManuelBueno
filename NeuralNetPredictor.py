from sklearn.base import BaseEstimator, RegressorMixin

class NeuralNetPredictor(BaseEstimator, RegressorMixin):
    """
    Custom scikit-learn predictor that delegates fit and predict to PreprocessData.
    """

    def __init__(self, neural_net=None):
        self.neural_net = neural_net

    def fit(self, X, y):
        self.neural_net.fit(X, y)
        return self

    def predict(self, X):
        y_aux = self.neural_net.predict(X)
        return y_aux

