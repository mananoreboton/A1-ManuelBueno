# activity1.py

from sklearn.datasets import fetch_california_housing

class Activity1:
    def __init__(self):
        """
        Initialize the Activity1 class.
        """
        print("Activity 1 library initialized.")

    def select_and_analyze_dataset(self):
        """
        Task 1: Fetch and analyze the California Housing dataset.
        """
        print("Fetching California Housing dataset...")
        data = fetch_california_housing(as_frame=True)
        df = data.frame

        print("\nDataset Description:")
        print(data.DESCR[:500])  # Display the first 500 characters of the dataset description

        print("\nDataset Preview (First 5 rows):")
        print(df.head())

        print("\nDataset Structure:")
        print(df.info())

        print("\nStatistical Summary:")
        print(df.describe())

        # Return the dataset as a DataFrame for further use
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
