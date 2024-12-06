from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class NeuralNetPredictor(BaseEstimator, ClassifierMixin):
    """
    Custom scikit-learn predictor that delegates fit and predict to PreprocessData.
    """

    def __init__(self, *, param=1, neural_net=None, score_type='mae'):
        self.score_type = score_type
        self.neural_net = neural_net
        self.param = param
        self.scores = {}

    def fit(self, X, y):
        self.neural_net.fit(X, y)
        return self

    def predict(self, X):
        y_aux = self.neural_net.predict(X)
        return y_aux

    def score(self, X, y, sample_weight=None):
        predictions = self.predict(X)
        self.scores['mse'] = mean_squared_error(y, predictions)
        self.scores['mae'] = mean_absolute_error(y, predictions)
        self.scores['mape'] = sum(abs((y_true - y_pred_val) / y_true) for y_true, y_pred_val in zip(y, predictions) if y_true != 0) / len(y)
        return self.scores[self.score_type]

    def get_scores(self):
        return self.scores
