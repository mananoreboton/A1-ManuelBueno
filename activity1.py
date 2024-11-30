# activity1.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from datetime import datetime

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

        return df[selected_features + [target_variable]]

    def transform_dates(self, df):
        """
        Subtask 1.2: Transform 'date' and 'yr_built' for better usability.
        """
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT000000')
        df['days_since_yr_built'] = (datetime.now() - pd.to_datetime(df['yr_built'], format='%Y')).dt.days
        df.drop(columns=['yr_built'], inplace=True)
        return df

    def transform_small_integers(self, df):
        """
        Subtask 1.3: Normalize 'bedrooms' and 'bathrooms' using MinMaxScaler.
        """
        scaler = MinMaxScaler()
        df[['bedrooms', 'bathrooms']] = scaler.fit_transform(df[['bedrooms', 'bathrooms']])
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
        Subtask 1.6: One-hot encode categorical features: 'waterfront', 'view', 'condition', 'grade'.
        """
        categorical_features = ['waterfront', 'view', 'condition', 'grade']
        encoder = OneHotEncoder(sparse=False, drop='first')  # Drop first to avoid multicollinearity
        encoded = encoder.fit_transform(df[categorical_features])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))
        df = df.drop(columns=categorical_features).reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)
        return df

    def preprocess_dataset(self, df):
        """
        Preprocess the dataset by applying all subtasks.
        """
        print("Preprocessing dataset...")
        df = self.select_features(df)  # Subtask 1.1
        df = self.transform_dates(df)  # Subtask 1.2
        df = self.transform_small_integers(df)  # Subtask 1.3
        df = self.transform_small_floats(df)  # Subtask 1.4
        df = self.transform_large_values(df)  # Subtask 1.5
        df = self.transform_categorical_values(df)  # Subtask 1.6
        print("Dataset preprocessing complete.")
        return df

    def select_and_analyze_dataset(self):
        """
        Task 1: Load, preprocess, and analyze the House Sales dataset.
        """
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
