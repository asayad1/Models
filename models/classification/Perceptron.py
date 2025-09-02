"""
This implements a classic Perceptron binary classifier using the perceptron update rule with a
unit step activation function. Training is done on the Iris dataset. 

Training supports learning multiple weights.
"""

import pandas as pd 
from typing import List
import matplotlib.pyplot as plt
import numpy as np

class Perceptron:
    weights: pd.Series = None
    bias: float = None 
    errors: List[int] = None
    epochs: int
    learning_rate: float 

    def __init__(self, *, learning_rate: float, epochs: int):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def net_input(self, *, X: np.ndarray):
        return X.dot(self.weights) + self.bias

    def threshold(self, *, net_input: float):
        return 1 if net_input >= 0 else 0

    def score(self, *, X: np.ndarray):
        return self.net_input(X=X)

    def fit(self, *, X: pd.DataFrame, y: pd.Series):
        # Initialize the weights and bias to 0 
        self.weights = pd.Series(0.0, index=X.columns)
        self.bias = 0
        
        for _ in range(self.epochs):
            # For each training example:
            # 1) Compute the output value: w1 x1 + w2 x2 + .. + wn xn + b
            # 2) Update the weights and bias unit
            self.errors = []

            for i in range(X.shape[0]):
                y_prediction = self.predict(X.iloc[i])
                error = y[i] - y_prediction
                self.weights += self.learning_rate * error * X.iloc[i]
                self.bias += self.learning_rate * error
                self.errors.append(abs(error))
            
            # See if we have converged early
            if not (1 in self.errors): break 

    def predict(self, X_i: pd.Series, *, with_score: bool = False) -> float:
        score = self.score(X=X_i)

        if not with_score:
            return self.threshold(net_input=score)
        else:
            return (score, self.threshold(net_input=score))


if __name__ == '__main__':
    dataset_url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    dataset = pd.read_csv(dataset_url)

    # Drop rows where species == 'virginica'
    dataset = dataset[dataset['species'] != 'virginica']

    # Features: drop species and the other columns
    X = dataset.drop(columns=['species', 'sepal_width', 'petal_width'], axis=1)

    # Labels: 1 if setosa, 0 otherwise
    y = (dataset['species'] == 'setosa').astype(int)

    # Train perceptron
    p = Perceptron(learning_rate=1e-2, epochs=500)
    p.fit(X=X, y=y)

    # Create mesh grid
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Predict for each grid point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for point in grid_points:
        Z.append(p.predict(X_i=point))

    Z = np.array(Z).reshape(xx.shape)

    # Plot filled decision regions
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.bwr)

    # Plot training points
    plt.scatter(
        X.iloc[:, 0], X.iloc[:, 1],
        c=['red' if label == 1 else 'blue' for label in y],
        edgecolor='k',
        marker='o'
    )

    plt.xlabel("Sepal length [cm]")
    plt.ylabel("Petal length [cm]")
    plt.title("Perceptron Decision Regions")
    plt.show()