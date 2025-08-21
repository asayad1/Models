"""
This implements a classic Perceptron using the perceptron update rule with a
unit step activation function. Training is done on the Iris dataset.
"""

import pandas as pd 
from typing import List

class Perceptron:
    weights: pd.Series = None
    bias: float = None 
    errors: List[int] = None

    def activation(self, activity: float):
        return 1 if activity >= 0 else 0 

    def fit(self, *, X: pd.DataFrame, y: pd.Series, learning_rate: float, epochs: int):
        # Initialize the weights and bias to 0 
        self.weights = pd.Series(0.0, index=X.columns)
        self.bias = 0
        
        for _ in range(epochs):
            # For each training example:
            # 1) Compute the output value: w1 x1 + w2 x2 + .. + wn xn + b
            # 2) Update the weights and bias unit
            self.errors = []

            for i in range(X.shape[0]):
                y_prediction = self.predict(X.iloc[i])
                error = y[i] - y_prediction
                self.weights += learning_rate * error * X.iloc[i]
                self.bias += learning_rate * error
                self.errors.append(abs(error))
            
            # See if we have converged early
            if not (1 in self.errors): break 


    def predict(self, X: pd.Series) -> float:
        activity = X.dot(self.weights) + self.bias
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