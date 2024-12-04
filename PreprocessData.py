# activity1.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
import joblib
import os

selected_features = ['date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                     'floors', 'waterfront', 'view', 'condition', 'grade',
                     'yr_built', 'lat', 'long', 'price']

class PreprocessData:
    def __init__(self):
        """
        Initialize PreprocessData.
        """
        print("PreprocessData initialized.")

    def main(self):
        """
        Main function to execute all subtasks of select and analyze dataset.
        """
        print("Executing all subtasks of select and analyze dataset...")

        # Task 1: Dataset Selection and Analysis
        self.select_and_analyze_dataset()

        # self.show_data(transformer.get_feature_names_out(), test_array_transformed, Y)

    def read_csv_file(self, filepath="./data/kc_house_data.csv"):
        """
        Subtask 1.1: Read the CSV file into a pandas DataFrame.
        """
        print(f"Reading data from {filepath}...")
        df = pd.read_csv(filepath)
        return df

    def truncate_dataframe(self, df, rows=2000):
        """
        Subtask 1.2: Limit the DataFrame to a specified number of rows.
        """
        print(f"Truncating data randomly to {rows} rows")
        return df.sample(n=rows, random_state=31).reset_index(drop=True)
    
    def filter_features(self, df, required_features=None):
        """
        Subtask 1.3: Keep only specified columns in the DataFrame.
        """
        if required_features is None:
            required_features = selected_features
        print(f"Selecting this columns from the data: {selected_features}")
        return df[required_features]
    
    def drop_missing_values(self, df, columns=None):
        """
        Subtask 1.4: Remove rows with missing values in the specified columns.
        """
        if columns is None:
            columns = selected_features
        print(f"Removing missing values from columns: {selected_features}")
        return df.dropna(subset=columns).reset_index(drop=True)

    def drop_outliers(self, df, columns=None):
        """
        Subtask 1.5: Remove rows with outliers in the specified columns using the IQR method.
        """
        if columns is None:
            columns = []
        df = df.copy()
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        print(f"Removing outliers values from columns: {columns}")
        return df.reset_index(drop=True)

    def split_data(self, df):
        """
        Subtask 1.6: Split data into training and test sets.
        """
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=31)
        print(f"Splitting data: train_data ({len(train_data.index)}) and test_data ({len(test_data.index)})")
        return train_data, test_data

    def create_column_transformer(self):
        """
        Subtask 1.7: Create a ColumnTransformer with preprocessing pipelines for each column.
        """
        # Pipelines for individual columns
        pipelines = {
            'date': make_pipeline(ConvertDateToDays(date_column='date'), MinMaxScaler()),
            'bedrooms': Pipeline([
                ('scale', MinMaxScaler()),
            ]),
            'bathrooms': Pipeline([
                ('scale', MinMaxScaler()),
            ]),
            'sqft_living': Pipeline([
                ('scale', MinMaxScaler()),
            ]),
            'sqft_lot': Pipeline([
                ('scale', MinMaxScaler()),
            ]),
            'floors': Pipeline([
                ('encode', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown="error")),
            ]),
            'waterfront': Pipeline([
                ('encode', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown="error")),
            ]),
            'view': Pipeline([
                ('encode', OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown="error")),
            ]),
            'condition': Pipeline([
                ('scale', MinMaxScaler()),
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
            ]),
            'price': Pipeline([
                ('scale', MinMaxScaler()),
            ])
        }

        # Combine all pipelines into a ColumnTransformer
        transformer = ColumnTransformer(
            transformers=[(col, pipelines[col], [col]) for col in pipelines],
            remainder='drop'  # Drop columns not explicitly specified
        )
        print(f"Creating ColumnTransformer")
        return transformer

    def fit_training_data(self, transformer, train_data):
        """
        Subtask 1.8: Adjust features in train data set
        """
        transformer.fit(train_data)
        print(f"Fit train_data")
        return transformer, train_data

    def transform_data(self, transformer, train_data, test_data):
        """
        Subtask 1.9: Apply transformations to features
        """
        train_data_transformed = transformer.transform(train_data)
        test_data_transformed = transformer.transform(test_data)
        print(f"Transforming train_data and test_data")
        return train_data_transformed, test_data_transformed

    def save_transformed_data(self, transformed_train_matrix, transformed_test_matrix, transformer, output_dir='./data'):
        """
         Subtask 1.9: Saves transformed train and test matrices as files
        """

        os.makedirs(output_dir, exist_ok=True)

        column_names = transformer.get_feature_names_out()
        train_df = pd.DataFrame(transformed_train_matrix, columns=column_names)
        test_df = pd.DataFrame(transformed_test_matrix, columns=column_names)

        # Save the matrices as CSV
        train_path = os.path.join(output_dir, "transformed_train_matrix.csv")
        test_path = os.path.join(output_dir, "transformed_test_matrix.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # Save the transformer as a .pkl file
        transformer_path = os.path.join(output_dir, "transformer.pkl")
        joblib.dump(transformer, transformer_path)

        print(f"Train matrix saved to {train_path}")
        print(f"Test matrix saved to {test_path}")
        print(f"Transformer saved to {transformer_path}")

    def select_and_analyze_dataset(self, size=2000):
        """
        Task 1: Load, preprocess, and analyze the House Sales dataset.
        """
        df = self.read_csv_file()
        df = self.truncate_dataframe(df, size)
        df = self.filter_features(df)
        df = self.drop_missing_values(df)
        df = self.drop_outliers(df)
        train_data, test_data = self.split_data(df)
        transformer = self.create_column_transformer()
        transformer, train_data = self.fit_training_data(transformer, train_data)
        transformed_train_matrix, transformed_test_matrix = self.transform_data(transformer, train_data, test_data)
        self.save_transformed_data(transformed_train_matrix, transformed_test_matrix, transformer)
        print("Executed all subtasks of select and analyze dataset...")

    def show_data(self, feature_names, array, columns_to_plot=None):
        if columns_to_plot is None:
            columns_to_plot = ['condition__condition_5', 'grade__grade', 'yr_built__yr_built', 'lat__lat', 'long__long','price']
        dfc = pd.DataFrame(array, columns=feature_names)

        pd.set_option('display.max_columns', None)
        print(feature_names)
        print(dfc.describe(percentiles=[.1, .2, .3, .6, .7, .8, .9, .999], include='all'))
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
        X = X.copy()
        self.min_date = pd.to_datetime(X[self.date_column], format='%Y%m%dT000000').min()
        return self

    def transform(self, x_to_transform):
        x = x_to_transform.copy()
        x[self.date_column] = pd.to_datetime(x[self.date_column], format='%Y%m%dT000000')
        x[f'days_since_first_{self.date_column}'] = (x[self.date_column] - self.min_date).dt.days
        x.drop(columns=[self.date_column], inplace=True)
        return x

    def get_feature_names_out(self, input_features=None):
        return [f'days_since_first_{self.date_column}']

if __name__ == "__main__":
    preprocessData = PreprocessData()
    preprocessData.main()
