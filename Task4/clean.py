import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


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



class XGBoostRegressorScratch:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def _gradient(self, y, y_pred):
        """
        Compute the gradient of the loss function (residuals).
        """
        return y - y_pred

    def fit(self, X, y):
        """
        Train the XGBoost model by sequentially adding trees.
        """
        # Initial prediction (mean of target)
        y_pred = np.mean(y)
        self.initial_prediction = y_pred
        y_pred = np.full(y.shape, self.initial_prediction)

        for _ in range(self.n_estimators):
            # Compute residuals (negative gradient)
            residuals = self._gradient(y, y_pred)

            # Fit a decision tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # Update predictions
            y_pred += self.learning_rate * tree.predict(X)

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        y_pred = np.full((X.shape[0],), self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
    

class RandomForestRegressorScratch:
    def __init__(self, n_estimators=10, max_features="sqrt", max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []

    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample from the dataset.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Train the Random Forest by fitting multiple Decision Trees.
        """
        X = np.array(X)  # Convert DataFrame to numpy array if needed
        y = np.array(y)  # Convert Series to numpy array if needed
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Random feature selection
            n_features = X.shape[1]
            max_features = int(np.sqrt(n_features)) if self.max_features == "sqrt" else n_features
            selected_features = np.random.choice(n_features, max_features, replace=False)
            
            # Fit the tree on selected features
            tree.fit(X_sample[:, selected_features], y_sample)
            self.trees.append((tree, selected_features))

    def predict(self, X):
        """
        Predict by averaging predictions from all trees.
        """
        X = np.array(X)  # Convert DataFrame to numpy array if needed
        
        tree_predictions = []
        for tree, features in self.trees:
            tree_predictions.append(tree.predict(X[:, features]))
        return np.mean(tree_predictions, axis=0)



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
    
    # Train XGBoost Regressor
    xgb = XGBoostRegressorScratch(n_estimators=100, learning_rate=0.1, max_depth=3)
    xgb.fit(X_train, y_train)
    
    # Train Random Forest
    rf = RandomForestRegressorScratch(n_estimators=10, max_depth=5)
    rf.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = xgb.predict(X_test)
    y_pred = rf.predict(X_test)
    
    # Evaluate the model
    mse = model.mean_squared_error(y_test, y_pred)
    r2 = model.r2_score(y_test, y_pred)

    print("Report of linear regression")
    print("Weights (Intercept and Coefficients):", model.weights.flatten())
    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)
    
    # for xgb
    print("MSE of XGBoostRegressorScratch")
    print("MSE:", np.mean((y_test - y_pred) ** 2))
    
    # for rf
    print("MSE of Random forest")
    print("MSE:", np.mean((y_test - y_pred) ** 2))

    
