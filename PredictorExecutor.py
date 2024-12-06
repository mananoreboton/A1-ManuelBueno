from NeuralNet import NeuralNet
from NeuralNetPredictor import NeuralNetPredictor
from PreprocessData import PreprocessData
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

class PredictorExecutor:
    def cross_validation(self, hyperparameters, X_in, y_in):
        hyperparameters_and_scores_by_index = {}

        for i, params in hyperparameters.iterrows():
            neural_net = NeuralNet(
                L=params["Number of Layers"],
                n=eval(params["Layer Structure"]),
                n_epochs=params["Num Epochs"],
                learning_rate=params["Learning Rate"],
                momentum=params["Momentum"],
                activation_function=params["Activation Function"],
                validation_split=0
            )

            neural_net_predictor = NeuralNetPredictor(neural_net=neural_net, scorer=mean_squared_error)
            scores = cross_val_score(neural_net_predictor, X_in, y_in, cv=10)
            hyperparameters_and_scores_by_index[i] = (params, scores)
        return hyperparameters_and_scores_by_index

if __name__ == "__main__":
    hyperparameters = pd.read_csv("./data/neural_network_parameters.csv")

    p = PreprocessData()
    X_in, y_in = p.read_transformed_data_from_file()
    X_in, y_in = X_in[:10], y_in[:10]

    predictor_executor = PredictorExecutor()
    hyperparameters_and_score_by_index = predictor_executor.cross_validation(hyperparameters, X_in, y_in)

    print(hyperparameters_and_score_by_index)
