"""
This implements a classic Perceptron using the perceptron update rule with a
unit step activation function. Training is done on the Iris dataset.
"""

import pandas as pd 
from typing import List

class Perceptron:
    weights: List[float] = None
    bias: float = None 

    def activation(self, activity: float):
        return 1 if activity >= 0 else 0 

    def fit(self, *, X: pd.DataFrame, y: pd.Series, learning_rate: float, epochs: int):
        # Initialize the weights and bias to 0 
        self.weights = [0] * X.shape[1]
        self.bias = 0

        for _ in range(epochs):
            # For each training example:
            # 1) Compute the output value: w1 x1 + w2 x2 + .. + wn xn + b
            # 2) Update the weights and bias unit
            for i in range(X.shape[0]):
                activity = 0
                for x in range(X.shape[1]):
                    activity += self.weights[x] * X.iloc[:, x][i]
                activity += self.bias
                
                y_prediction = self.activation(activity)
                
                for x in range(X.shape[1]):
                    self.weights[x] += learning_rate * (y[i] - y_prediction) * X.iloc[:, x][i]
                self.bias += learning_rate * (y[i] - y_prediction)


    def predict(self, X: pd.DataFrame):
        activity = 0
        for x in range(X.shape[1]):
            activity += self.weights[x] * X.iloc[:, x][0]
        activity += self.bias

    
        y_prediction = self.activation(activity)
        return y_prediction
            


if __name__ == '__main__':
    dataset_url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    dataset = pd.read_csv(dataset_url)
    
    X = dataset.drop('species', axis=1)
    y = (dataset['species'] == 'setosa').astype(int)

    p = Perceptron()
    p.fit(X=X, y=y, learning_rate=1, epochs=500)

    print(p.weights, p.bias)