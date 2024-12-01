# activity1.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from scipy.stats import zscore

class Activity1:
    def __init__(self):
        """
        Initialize the Activity1 class.
        """
        print("Activity 1 library initialized.")

    def select_features(self, df):
        """
        Subtask 1.1: Select relevant features for the analysis.
        """
        selected_features = [
            'date', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
            'floors', 'waterfront', 'view', 'condition', 'grade', 
            'yr_built', 'lat', 'long'
        ]
        target_variable = 'price'

        if target_variable not in df.columns:
            raise ValueError("The target variable 'price' is not in the dataset.")

        if any(feature not in df.columns for feature in selected_features):
            raise ValueError("One or more selected features are missing from the dataset.")

        # Return a copy of the selected columns
        return df[selected_features + [target_variable]].copy()

    def transform_dates(self, df):
        """
        Subtask 1.2: Transform 'date' and 'yr_built' for better usability.
        """
        df = df.copy()
        # We consider that separating the date values ​​is not convenient because they are 
        # not expected to influence the price of the houses separately (e.g. Seasonal variations).
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT000000')
        df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

        # Normalize 'days_since_start' and 'yr_built'
        scaler = MinMaxScaler()
        df[['days_since_start', 'yr_built']] = scaler.fit_transform(df[['days_since_start', 'yr_built']])
        # Drop the original 'date' column
        df.drop(columns=['date'], inplace=True)
        return df

    def transform_small_integers(self, df):
        """
        Subtask 1.3: Normalize 'bedrooms', 'bathrooms', and 'grade' using MinMaxScaler.
        """
        df = df.copy()
        scaler = MinMaxScaler()
        df[['bedrooms', 'bathrooms', 'grade']] = scaler.fit_transform(df[['bedrooms', 'bathrooms', 'grade']])
        return df

    def transform_small_floats(self, df):
        """
        Subtask 1.4: Normalize 'floors' (small float values) using MinMaxScaler.
        """
        scaler = MinMaxScaler()
        df[['floors']] = scaler.fit_transform(df[['floors']])
        return df

    def transform_large_values(self, df):
        """
        Subtask 1.5: Standardize 'sqft_living' and 'sqft_lot' using StandardScaler.
        """
        scaler = StandardScaler()
        df[['sqft_living', 'sqft_lot']] = scaler.fit_transform(df[['sqft_living', 'sqft_lot']])
        return df

    def transform_categorical_values(self, df):
        """
        Subtask 1.6: One-hot encode categorical features: 'waterfront', 'view', 'condition'.
        """
        df = df.copy()
        categorical_features = ['waterfront', 'view', 'condition']
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded = encoder.fit_transform(df[categorical_features])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))
        df = df.drop(columns=categorical_features).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def normalize_coordinates(self, df):
        """
        Subtask 1.7: Normalize 'lat' and 'long' using MinMaxScaler.
        """
        df = df.copy()

        # Normalize 'lat' and 'long'
        scaler = MinMaxScaler()
        df[['lat', 'long']] = scaler.fit_transform(df[['lat', 'long']])
        return df
    
    def reduce_sample_size(self, df, sample_size=2000, r_state=31):
        """
        Subtask 1.8: Reduce the sample size to the specified number of rows (random selection).
        """
        df = df.sample(n=sample_size, random_state=r_state).reset_index(drop=True)
        print(f"Reduced dataset to {len(df)} rows.")
        return df
    
    def check_missing_values(self, df):
        """
        Subtask 1.9: Check for missing values in the dataset.
        """
        missing = df.isnull().sum()
        print("\nMissing values per column:")
        print(missing[missing > 0])
        
        # Drop rows with missing values
        if missing.any():
            df = df.dropna().reset_index(drop=True)
            print(f"Dataset size after removing missing values: {len(df)} rows.")
        return df
    
    def detect_outliers(self, df, threshold=3.0):
        """
        Subtask 1.10: Detect and optionally handle outliers using z-scores.
        """

        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        z_scores = zscore(df[numeric_cols])
        outliers = (abs(z_scores) > threshold).any(axis=1)

        print(f"Found {outliers.sum()} outliers out of {len(df)} rows.")
        
        df = df[~outliers].reset_index(drop=True)
        print(f"Dataset size after removing outliers: {len(df)} rows.")
        return df
    
    def preprocess_dataset(self, df):
        """
        Preprocess the dataset by applying all subtasks.
        """
        print("Preprocessing dataset...")
        df = self.check_missing_values(df)  # Subtask 1.9
        df = self.reduce_sample_size(df)  # Subtask 1.8
        df = self.select_features(df)  # Subtask 1.1
        df = self.transform_dates(df)  # Subtask 1.2
        df = self.transform_small_integers(df)  # Subtask 1.3
        df = self.transform_small_floats(df)  # Subtask 1.4
        df = self.transform_large_values(df)  # Subtask 1.5
        df = self.transform_categorical_values(df)  # Subtask 1.6
        df = self.normalize_coordinates(df)  # Subtask 1.7
        df = self.detect_outliers(df)  # Subtask 1.10
        print("Dataset preprocessing complete.")
        return df

    def select_and_analyze_dataset(self):
        """
        Task 1: Load, preprocess, and analyze the House Sales dataset.
        """
        pd.set_option('display.max_columns', None)

        print("Loading House Sales Prediction dataset locally...")
        dataset_path = "./kc_house_data.csv"  # Path to the CSV file (ensure this file exists in the project directory)

        # Load the dataset into a DataFrame
        df = pd.read_csv(dataset_path)

        print("\nDataset Preview (First 5 rows):")
        print(df.head())

        print("\nDataset Structure:")
        print(df.info())

        # Preprocess the dataset
        preprocessed_df = self.preprocess_dataset(df)

        print("\nPreprocessed Dataset Preview (First 5 rows):")
        print(preprocessed_df.head())

        return preprocessed_df

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

    def main(self):
        """
        Main function to execute all tasks.
        """
        print("Starting Activity 1 tasks...")
        
        # Task 1: Dataset Selection and Analysis
        dataset = self.select_and_analyze_dataset()

        # Additional tasks can be called here
        self.implement_neural_network_bp()
        self.implement_multiple_linear_regression()
        self.implement_neural_network_bp_f()

        print("All tasks executed. Add functionality to individual methods as needed.")

# If this script is executed directly, demonstrate the class functionality.
if __name__ == "__main__":
    activity1 = Activity1()
    activity1.main()
