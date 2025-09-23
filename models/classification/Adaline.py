"""
This implements Adaline using the least mean squares (LMS) rule with a
linear activation function. Training is done on the Iris dataset.

Adaline attempts to learn a linear decision boundary between two classes by modifying its weight and bias.
The change in weight and bias is proportional to its gradient with respect to the loss function (minimizing the loss). 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    weights: pd.Series
    bias: float
    epochs: int
    learning_rate: float 
    lmbda: float
    regularization = ['l2', 'l1', 'none']

    def __init__(self, *, learning_rate: float, epochs: int, lmbda: float = 0.0, regularization: str = 'none'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lmbda = lmbda
        self.regularization = regularization

    def net_input(self, *, X: np.ndarray):
        return X.dot(self.weights) + self.bias

    def activation(self, *, z: float):
        return z

    def threshold(self, *, activation: float):
        return 1 if activation >= 0.5 else 0

    def score(self, X):
        return self.activation(z=self.net_input(X=X))

    def fit(self, *, X: pd.DataFrame, y: pd.Series):
        self.weights = np.full(X.shape[1], 0.01)
        self.bias = 0.01

        for _ in range(self.epochs):
            delta_w = []
            delta_b = 0
            for X_i, y_i in zip(X.values, y):
                # Compute net input function 
                y_hat = self.activation(z=self.net_input(X=X_i))
                
                # Compute the error
                error = y_i - y_hat
                delta_w.append(error * X_i)
                delta_b += error
            
            # Average out the error for full batch GD
            delta_w = [(-2 / X.shape[0] * sum(i)) for i in zip(*delta_w)]
            delta_b = -2 / X.shape[0] * delta_b
            
            # Apply weight updates
            if self.regularization == 'l1':
                delta_w += self.lmbda * np.sign(self.weights)
            elif self.regularization == 'l2':
                delta_w += self.lmbda * self.weights
                
            self.weights -= self.learning_rate * np.array(delta_w)
            self.bias -= self.learning_rate * delta_b

    def predict(self, X_i: pd.Series, *, with_score: bool = False) -> float:
        score = self.score(X=X_i)

        if not with_score:
            return self.threshold(activation=score)
        else:
            return (score, self.threshold(activation=score))


if __name__ == '__main__':
    dataset_url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    dataset = pd.read_csv(dataset_url)

    # Drop rows where species == 'virginica'
    dataset = dataset[dataset['species'] != 'virginica']

    # Features: drop species and the other columns
    X = dataset.drop(columns=['species', 'sepal_width', 'petal_width'], axis=1)

    # Labels: 1 if setosa, 0 otherwise
    y = (dataset['species'] == 'setosa').astype(int)

    # Define the model
    model = Adaline(learning_rate=1e-3, epochs=1000)
    model.fit(X=X, y=y)

    # Create mesh grid
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    # Predict for each grid point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for point in grid_points:
        Z.append(model.threshold(activation=model.activation(z=model.net_input(X=point))))
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
    plt.title("Adaline Decision Regions")
    plt.show()