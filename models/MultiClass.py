"""
This file implements an abstract multi-class classifier from binary classifiers.

Strategies include: One-Versus-All/Rest (OvA/Ovr), OvO (One-Versus-One).

In OvA, a single classifier is trained for each class, distinguishing it from all other classes.
In OvO, a classifier is trained for every pair of classes.
"""

import pandas as pd
from Perceptron import Perceptron
from Adaline import Adaline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np 

class MultiClass:
    model = None            # The binary classifer we are using
    strategy: str = None    # The strategy we are using for multiclass
    classifiers: dict
    epochs: int
    learning_rate: float
    X: pd.DataFrame
    y: pd.Series
    
    def __init__(self, *, model, strategy: str, X: pd.DataFrame, y: pd.Series, epochs: int, learning_rate: float):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.X = X
        self.y = y 

        # Validate strategy
        if strategy.lower() in ['ova', 'ovo']: 
            self.strategy = strategy.lower()
        else:
            raise ValueError("Strategy is not OvA or OvO!")
    
        # Determine the number of classifiers to create
        self.classifiers = {label: model(epochs=self.epochs, learning_rate=self.learning_rate) for label in set(self.y)}

    def OvA(self):
        for label in set(self.y):
            y = (self.y == label).astype(int)            
            self.classifiers[label].fit(X=self.X, y=y)

    def OvO(self):
        pass

    def fit(self):
        if self.strategy == 'ova':
            self.OvA()
        else:
            self.OvO()

    def predict(self, X_i: pd.Series):
        predictions = {label: self.classifiers[label].predict(X_i, with_activation=True) for label in self.classifiers}    
        return max(predictions, key=predictions.get)

if __name__ == '__main__':
    # Load Iris
    dataset_url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
    dataset = pd.read_csv(dataset_url)

    # Use two features for visualization purposes
    feat_cols = ['sepal_length', 'petal_length']
    X = dataset[feat_cols].copy()
    y = dataset['species'].copy()

    # Standardize (z-score)
    means = X.mean()
    stds = X.std(ddof=0)
    X_std = (X - means) / stds

    # Train OvA
    model = MultiClass(model=Adaline, strategy='OvA', X=X_std, y=y, epochs=10000, learning_rate=1e-2)
    model.fit()


    # Decision region grid
    x1_min, x1_max = X_std[feat_cols[0]].min() - 1.0, X_std[feat_cols[0]].max() + 1.0
    x2_min, x2_max = X_std[feat_cols[1]].min() - 1.0, X_std[feat_cols[1]].max() + 1.0

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 400),
        np.linspace(x2_min, x2_max, 400)
    )

    grid = np.c_[xx1.ravel(), xx2.ravel()]
    grid_df = pd.DataFrame(grid, columns=feat_cols)

    # Class label order & mapping for colors
    labels = sorted(y.unique())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    # Predict each grid point with MultiClass model
    Z_labels = []
    for i in range(grid_df.shape[0]):
        Z_labels.append(model.predict(grid_df.iloc[i, :]))
    Z_idx = np.array([label_to_idx[z] for z in Z_labels]).reshape(xx1.shape)

    # Three pleasant, distinct background colors
    cmap_bg = ListedColormap(['#f6a5a5', '#9fb7ff', '#c7f0c2'])
    cmap_pts = ListedColormap(['#b30000', '#0033cc', '#1a7f37'])

    plt.figure(figsize=(10, 6))
    plt.contourf(xx1, xx2, Z_idx, alpha=0.35, cmap=cmap_bg)

    # Plot the standardized points per class with distinct markers
    markers = ['o', 's', '^']
    for i, lab in enumerate(labels):
        Xi = X_std[y == lab]
        plt.scatter(
            Xi[feat_cols[0]].values, Xi[feat_cols[1]].values,
            c=[cmap_pts(i)],
            marker=markers[i % len(markers)],
            edgecolor='k',
            label=f'Class {i} ({lab})'
        )

    plt.xlabel(f'{feat_cols[0]} [standardized]')
    plt.ylabel(f'{feat_cols[1]} [standardized]')
    plt.legend(loc='upper left', framealpha=0.95)
    plt.title('OvA decision regions (Adaline)')
    plt.tight_layout()
    plt.show()