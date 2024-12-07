from NeuralNet import NeuralNet
from NeuralNetPredictor import NeuralNetPredictor
from PreprocessData import PreprocessData
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

class PredictorExecutor:
    def save_cross_validation_score_list(self, case, scores, scoring):
        scores_path = "./data/cross_validation_with_hyperparameters_" + str(case) + "_" + scoring + ".csv"
        np.savetxt(scores_path, scores, delimiter=',', fmt='%.9f')
        print(f"Saving Cross Validation ccore list for case '{case}' in '{scores_path}'")

    def cross_validation_for_this_hyperparameter(self, hyperparameters, X, y, scoring='neg_mean_absolute_error', folds=10):
        neural_net = NeuralNet(
            L=hyperparameters["Number of Layers"],
            n=eval(hyperparameters["Layer Structure"]),
            n_epochs=hyperparameters["Num Epochs"],
            learning_rate=hyperparameters["Learning Rate"],
            momentum=hyperparameters["Momentum"],
            activation_function=hyperparameters["Activation Function"],
            validation_split=0
        )

        neural_net_predictor = NeuralNetPredictor(neural_net=neural_net)
        scores = cross_val_score(neural_net_predictor, X, y, cv=folds, scoring=scoring)
        return scores, np.mean(scores), np.var(scores)

    def cross_validation(self, hyperparameters, X, y, scoring, folds):
        scores_by_hyperparameters = []

        for i, hyperparameters in hyperparameters.iterrows():
            scores, mean, var = self.cross_validation_for_this_hyperparameter(hyperparameters, X, y, scoring, folds)
            self.save_cross_validation_score_list(i, scores, scoring)
            df = hyperparameters.copy()
            df['Scores'] = scores
            df['Mean'] = mean
            df['Variance'] = var
            scores_by_hyperparameters.append(df)
        return scores_by_hyperparameters

if __name__ == "__main__":
    hyperparameters = pd.read_csv("./data/neural_network_parameters.csv")

    p = PreprocessData()
    X_in, y_in = p.read_transformed_data_from_file()
    X_in, y_in = X_in[:1000], y_in[:1000]

    predictor_executor = PredictorExecutor()
    scores_by_hyperparameters = predictor_executor.cross_validation(
        hyperparameters,
        X_in,
        y_in,
        scoring='neg_mean_absolute_error',
        folds=10
    )
    print(scores_by_hyperparameters)

    predictor_executor = PredictorExecutor()
    scores_by_hyperparameters = predictor_executor.cross_validation(
        hyperparameters,
        X_in,
        y_in,
        scoring='neg_mean_squared_error',
        folds=10
    )
    print(scores_by_hyperparameters)

    predictor_executor = PredictorExecutor()
    scores_by_hyperparameters = predictor_executor.cross_validation(
        hyperparameters,
        X_in,
        y_in,
        scoring='neg_mean_absolute_percentage_error',
        folds=10
    )
    print(scores_by_hyperparameters)
