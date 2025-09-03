"""
This file implements a logistic regression classifier for binary classification tasks.
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

class LogisticRegression:
    weights: pd.Series
    bias: float
    learning_rate: float 
    epochs: int

    def __init__(self, *, learning_rate: float, epochs: int):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def net_input(self, *, X: pd.Series):
        return X.dot(self.weights) + self.bias

    def activation(self, *, z: float):
        return 1 / (1 + np.exp(-z))

    def threshold(self, *, activation: float):
        return 1 if activation >= 0.5 else 0

    def fit(self, *, X: pd.DataFrame, y: pd.Series):
        self.weights = np.full(X.shape[1], 0.01)
        self.bias = 0

        for _ in range(self.epochs):
            delta_w = []
            delta_b = 0
            for X_i, y_i in zip(X.values, y):
                probability = self.activation(z=self.net_input(X=X_i))
                error = y_i - probability
                delta_w.append(error * X_i)
                delta_b += error

            # Average out the error for full batch GD
            delta_w = [(2 / X.shape[0] * sum(i)) for i in zip(*delta_w)]
            delta_b = 2 / X.shape[0] * delta_b
            
            self.weights += self.learning_rate * np.array(delta_w)
            self.bias += self.learning_rate * delta_b


    def predict(self, X_i: pd.Series):
        return self.threshold(activation=self.activation(z=self.net_input(X=X_i)))


if __name__ == '__main__':
    # Load the dataset
    dataset_url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    dataset = pd.read_csv(dataset_url)

    # Preprocess the dataset 
    features = ['sepal_length', 'petal_length']
    target = 'species'
    X = dataset[features]
    y = (dataset[target] == 'setosa').astype(int)

    # Instantiate Logistic Regression Classifier
    model = LogisticRegression(learning_rate=1e-2, epochs=500)
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
    plt.title("Logistic Regression Decision Regions")
    plt.show()