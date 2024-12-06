from NeuralNet import NeuralNet
from NeuralNetPredictor import NeuralNetPredictor
from PreprocessData import PreprocessData
from sklearn.model_selection import cross_val_score
import pandas as pd


if __name__ == "__main__":
    p = PreprocessData()
    X_in, y_in = p.read_transformed_data_from_file()

    hyperparameters_and_score_by_index = {}
    hyperparameters = pd.read_csv("./data/neural_network_parameters.csv")
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

        neural_net_predictor = NeuralNetPredictor(neural_net=neural_net)
        mae = cross_val_score(neural_net_predictor, X_in, y_in, cv=10)
        hyperparameters_and_score_by_index[i] = params
    print(hyperparameters_and_score_by_index)
