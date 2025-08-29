import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Adaline:
    weights: pd.Series
    bias: float

    def net_input(self, *, X: np.ndarray):
        return X.dot(self.weights) + self.bias

    def activation(self, *, z: float):
        return z

    def threshold(self, *, activation: float):
        return 1 if activation >= 0.5 else 0

    def fit(self, *, X: pd.DataFrame, y: pd.Series, learning_rate: float, iterations: int = 50000):
        self.weights = np.full(X.shape[1], 0.01)
        self.bias = 0.01

        for _ in range(iterations):
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
            self.weights -= learning_rate * np.array(delta_w)
            self.bias -= learning_rate * delta_b

        print(self.weights, self.bias)

    def predict(self, X_i: pd.Series):
        y_hat = self.threshold(activation=self.activation(z=self.net_input(X=X_i)))
        return y_hat

if __name__ == '__main__':
    dataset_url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    dataset = pd.read_csv(dataset_url)

    # Drop rows where species == 'virginica'
    dataset = dataset[dataset['species'] != 'virginica']

    # Features: drop species and the other columns
    X = dataset.drop(columns=['species', 'sepal_width', 'petal_width'], axis=1)
    # X = (X - X.mean()) / X.std()

    # Labels: 1 if setosa, 0 otherwise
    y = (dataset['species'] == 'setosa').astype(int)

    # Define the model
    model = Adaline()
    model.fit(X=X, y=y, learning_rate=1e-3)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        X["sepal_length"], X["petal_length"],
        c=['blue' if label == 1 else 'red' for label in y],
        edgecolors="k"
    )

    # Plot decision boundary: w1*x1 + w2*x2 + b = 0 → x2 = -(w1/w2)*x1 - b/w2
    x_vals = np.linspace(X["sepal_length"].min(), X["sepal_length"].max(), 100)
    w1, w2 = model.weights
    b = model.bias
    y_vals = -(w1 / w2) * x_vals - (b / w2)

    plt.plot(x_vals, y_vals, 'k--', label="Decision boundary")
    plt.xlabel("sepal_length")
    plt.ylabel("petal_length")
    plt.legend()
    plt.title("Adaline Decision Boundary on Iris Dataset")
    plt.show()