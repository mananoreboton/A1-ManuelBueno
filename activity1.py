# activity1.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

selected_features = ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                     'floors', 'waterfront', 'view', 'condition', 'grade',
                     'yr_built', 'lat', 'long', 'price']

class Activity1:
    def __init__(self):
        """
        Initialize Activity1.
        """

        print("Activity 1 initialized.")

    def main(self):
        """
        Main function to execute all tasks.
        """
        print("Starting Activity 1 tasks...")

        # Task 1: Dataset Selection and Analysis
        transformer, train_array_transformed, test_array_transformed, Y = self.select_and_analyze_dataset()

        # self.show_data(transformer.get_feature_names_out(), test_array_transformed, Y)

        # Additional tasks
        self.implement_neural_network_bp()
        self.implement_multiple_linear_regression()
        self.implement_neural_network_bp_f()

        print("All tasks executed. Add functionality to individual methods as needed.")

    def read_csv_file(self, filepath="./kc_house_data.csv"):
        """
        Subtask 1.1: Read the CSV file into a pandas DataFrame.
        """
        print(f"Reading data from {filepath}...")
        df = pd.read_csv(filepath)
        return df, df['price']

    def truncate_dataframe(self, df, rows=2000):
        """
        Subtask 1.2: Limit the DataFrame to a specified number of rows.
        """
        return df.sample(n=rows, random_state=31).reset_index(drop=True)
    
    def filter_features(self, df, required_features=None):
        """
        Subtask 1.3: Keep only specified columns in the DataFrame.
        """
        if required_features is None:
            required_features = selected_features
        return df[required_features]
    
    def drop_missing_values(self, df, columns=None):
        """
        Subtask 1.4: Remove rows with missing values in the specified columns.
        """
        if columns is None:
            columns = selected_features
        return df.dropna(subset=columns).reset_index(drop=True)

    def drop_outliers(self, df, columns = []):
        """
        Subtask 1.5: Remove rows with outliers in the specified columns using the IQR method.
        """
        df = df.copy()
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df.reset_index(drop=True)

    def split_data(self, df):
        """
        Subtask 1.6: Split data into training and test sets.
        """
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=31)
        return train_data, test_data

    def create_column_transformer(self):
        """
        Subtask 1.7: Create a ColumnTransformer with preprocessing pipelines for each column.
        """
        # Pipelines for individual columns
        pipelines = {
            'date': Pipeline([
                ('convert_date_to_days', ConvertDateToDays(date_column='date')),
            ]),
            'bedrooms': Pipeline([
                ('scale', MinMaxScaler()),
            ]),
            'bathrooms': Pipeline([
                ('scale', MinMaxScaler()),
            ]),
            'sqft_living': Pipeline([
                ('scale', StandardScaler()),
            ]),
            'sqft_lot': Pipeline([
                ('scale', RobustScaler()),
            ]),
            'floors': Pipeline([
                ('encode', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown="ignore")),
            ]),
            'waterfront': Pipeline([
                ('encode', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown="ignore")),
            ]),
            'view': Pipeline([
                ('encode', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown="ignore")),
            ]),
            'condition': Pipeline([
                ('encode', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown="ignore")),
            ]),
              'grade': Pipeline([
                ('scale', MinMaxScaler()),
            ]),
            'yr_built': Pipeline([
                ('scale', MinMaxScaler()),
            ]),
            'lat': Pipeline([
                ('scale', MinMaxScaler()),
            ]),
            'long': Pipeline([
                ('scale', MinMaxScaler()),
            ])
        }

        # Combine all pipelines into a ColumnTransformer
        transformer = ColumnTransformer(
            transformers=[(col, pipelines[col], [col]) for col in pipelines],
            remainder='drop'  # Drop columns not explicitly specified
        )
        return transformer

    def fit_training_data(self, transformer, train_data):
        """
        Subtask 1.8: Adjust features in train data set
        """
        transformer.fit(train_data)
        return transformer, train_data

    def transform_data(self, transformer, train_data, test_data):
        """
        Subtask 1.9: Apply transformations to features
        """
        train_data_transformed = transformer.transform(train_data)
        test_data_transformed = transformer.transform(test_data)
        return train_data_transformed, test_data_transformed


    def select_and_analyze_dataset(self):
        """
        Task 1: Load, preprocess, and analyze the House Sales dataset.
        """
        df, Y = self.read_csv_file()
        df = self.truncate_dataframe(df)
        df = self.filter_features(df)
        df = self.drop_missing_values(df)
        # df = self.drop_outliers(df)
        train_data, test_data = self.split_data(df)
        transformer = self.create_column_transformer()
        transformer, train_data = self.fit_training_data(transformer, train_data)
        train_array_transformed, test_array_transformed = self.transform_data(transformer, train_data, test_data)
        return transformer, train_array_transformed, test_array_transformed, Y

    def implement_neural_network_bp(self):
        """
        Task 2: Implement a neural network with back-propagation manually.
        """
        print("Implementing Neural Network with Back-Propagation (BP)... (To be implemented)")
    def implement_multiple_linear_regression(self):
        """
        Task 3: Implement Multiple Linear Regression using sklearn.
        """
        print("Implementing Multiple Linear Regression (MLR-F)... (To be implemented)")

    def implement_neural_network_bp_f(self):
        """
        Task 4: Implement a neural network with back-propagation using a library.
        """
        print("Implementing Neural Network with Back-Propagation (BP-F)... (To be implemented)")

    def show_data(self, feature_names, array, Y):
        dfc = pd.DataFrame(array, columns=feature_names)
        dfc['price'] = Y

        pd.set_option('display.max_columns', None)
        print(feature_names)
        print(dfc.describe(percentiles=[.1, .2, .3, .6, .7, .8, .9, .999], include='all'))
        columns_to_plot = ['condition__condition_5', 'grade__grade', 'yr_built__yr_built', 'lat__lat', 'long__long', 'price']
        df_subset = dfc[columns_to_plot]
        scatter_matrix(df_subset, figsize=(10, 10), alpha=0.8, diagonal='hist')
        plt.show()


class ConvertDateToDays(BaseEstimator, TransformerMixin):
    """
    Custom transformer to convert a date column into the number of days
    since the earliest date in the dataset.
    """
    def __init__(self, date_column='date'):
        self.date_column = date_column
        self.min_date = None

    def fit(self, X, y=None):
        # Store the minimum date during fit
        X = X.copy()
        self.min_date = pd.to_datetime(X[self.date_column], format='%Y%m%dT000000').min()
        return self

    def transform(self, X):
        # Convert dates to days since the minimum date
        X = X.copy()
        X[self.date_column] = pd.to_datetime(X[self.date_column], format='%Y%m%dT000000')
        X[f'days_since_{self.date_column}'] = (X[self.date_column] - self.min_date).dt.days
        X.drop(columns=[self.date_column], inplace=True)  # Drop the original date column
        return X

    def get_feature_names_out(self, input_features=None):
        return [f"{self.date_column}_publication"]
    
# If this script is executed directly, demonstrate the class functionality.
if __name__ == "__main__":
    activity1 = Activity1()
    activity1.main()
