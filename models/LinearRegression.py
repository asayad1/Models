"""
An ordinary least-squares linear regression model.

LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize 
the residual sum of squares between the observed targets in the dataset, and the 
targets predicted by the linear approximation. 

It uses batch gradient descent, with MSE as the loss function.
"""

from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression:
    coefficients: List[float] = None
    X: pd.DataFrame = None
    y: pd.Series = None
    prev_loss: float = None

    def MSE(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.coefficients[0] * X + self.coefficients[1]
        mse = ((y - y_pred) ** 2).mean()

        return mse

    def compute_gradients(self, *, X: pd.DataFrame, y: pd.Series):
        y_pred = self.coefficients[0] * X + self.coefficients[1]
        slope_gradient = -2 * (X * (y - y_pred)).mean()
        intercept_gradient = -2 * (y - y_pred).mean()

        return (slope_gradient, intercept_gradient)

    def fit(self, *, X: pd.DataFrame, y: pd.Series, learning_rate: float, max_iter: int = 50000, tol: float = 1e-3):
        # Initialize the coefficients list 
        num_features = 2
        self.coefficients = [0] * num_features
        
        
        for i in range(max_iter):
            # Calculate MSE 
            mse = self.MSE(X, y)
            
            # Determine if we have reached stopping criterion
            if (tol and self.prev_loss) and (abs(mse - self.prev_loss) <= tol):
                break
            
            # Compute gradients
            gradients = self.compute_gradients(X=X, y=y)

            # Update parameters
            self.coefficients[0] -= learning_rate * gradients[0]
            self.coefficients[1] -= learning_rate * gradients[1]

            self.prev_loss = mse 
        return self.coefficients

    def predict(self, X: float):
        return self.coefficients[0] * X + self.coefficients[1]


if __name__ == "__main__":
    data = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
    model = LinearRegression()
    
    model.fit(X=data['Experience Years'], y=data['Salary'], learning_rate=1e-3)
    print(f'y = {model.coefficients[0]} x + {model.coefficients[1]}')

    plt.scatter(data['Experience Years'], data['Salary'], label='Data')
    x_range = [data['Experience Years'].min(), data['Experience Years'].max()]
    y_pred = [model.predict(x) for x in x_range]
    plt.plot(x_range, y_pred, color='red', label='Regression Line')
    plt.xlabel('Experience Years')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()
    