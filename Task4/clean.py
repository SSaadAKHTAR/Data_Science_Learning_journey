import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Linear Regression Model
class LinearRegression:
    def __init__(self):
        self.weights = None  # To store the calculated weights

    def fit(self, X, y):
        """
        Train the model using the Normal Equation.
        """
        # Add a bias (intercept) term to X
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of 1s for the intercept
        
        # Calculate weights using the Normal Equation
        self.weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        # Add bias (intercept) term to X
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Compute predictions
        return X_bias @ self.weights

    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate Mean Squared Error (MSE).
        """
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        """
        Calculate R² score.
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


if __name__ == "__main__":
    data_path = 'Task4/BostonHousing.csv'
    raw_data = pd.read_csv(data_path)

    raw_data['rm'].fillna(raw_data['rm'].mean(), inplace=True)
    raw_data['high_crime'] = (raw_data['crim'] > raw_data['crim'].mean()).astype(int)
    
    X = raw_data.drop(columns=['medv','high_crime'])
    y = raw_data['medv']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = model.mean_squared_error(y_test, y_pred)
    r2 = model.r2_score(y_test, y_pred)

    print("Weights (Intercept and Coefficients):", model.weights.flatten())
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)

    
