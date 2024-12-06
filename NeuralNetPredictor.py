from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

class NeuralNetPredictor(BaseEstimator, ClassifierMixin):
    """
    Custom scikit-learn predictor that delegates fit and predict to PreprocessData.
    """

    def __init__(self, *, param=1, neural_net=None, scorer=mean_absolute_error):
        self.scorer = scorer
        self.neural_net = neural_net
        self.param = param

    def fit(self, X, y):
        self.neural_net.fit(X, y)
        return self

    def predict(self, X):
        y_aux = self.neural_net.predict(X)
        return y_aux

    def score(self, X, y, sample_weight=None):
        predictions = self.predict(X)
        return self.scorer(y, predictions)
