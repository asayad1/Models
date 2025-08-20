"""
This implements a classic Perceptron using the perceptron update rule with a
unit step activation function. Training is done on the Iris dataset.
"""

import pandas as pd 
from typing import List

class Perceptron:
    weights: List[float] = None

    def __init__(self):
        pass


    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Initialize the weights and bias to 0 
    
        # For each training example:
        # 1) Compute the output value: w1 x1 + w2 x2 + .. + wn xn + b
        # 2) Update the weights and bias unit

        pass

    def predict(self):
        pass


if __name__ == '__main__':
    dataset_url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    dataset = pd.read_csv(dataset_url)

    for i in dataset:
        print(i)
    print(dataset.head())