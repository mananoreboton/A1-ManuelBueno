# activity1.py

import pandas as pd

class Activity1:
    def __init__(self):
        """
        Initialize the Activity1 class.
        """
        print("Activity 1 library initialized.")

    def select_and_analyze_dataset(self):
        """
        Task 1: Load and analyze the House Sales dataset from a local file.
        """
        print("Loading House Sales Prediction dataset locally...")
        dataset_path = "./kc_house_data.csv"  # Path to the CSV file (ensure this file exists in the project directory)

        # Load the dataset into a DataFrame
        df = pd.read_csv(dataset_path)

        # Display the first few rows of the dataset
        print("\nDataset Preview (First 5 rows):")
        print(df.head())

        # Display dataset information
        print("\nDataset Structure:")
        print(df.info())

        # Display statistical summary
        print("\nStatistical Summary:")
        print(df.describe())

        return df

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
