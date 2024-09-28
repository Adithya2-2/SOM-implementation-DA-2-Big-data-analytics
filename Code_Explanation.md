## Table of Contents

- [Project Overview](#project-overview)
- [Libraries and Dependencies](#libraries-and-dependencies)
- [Dataset](#dataset)
- [Code Explanation](#code-explanation)
  - [Importing the Libraries](#1-importing-the-libraries)
  - [Loading the Dataset](#2-loading-the-dataset)
  - [Feature Scaling](#3-feature-scaling)
  - [Training the SOM](#4-training-the-som)
  - [Visualizing the Results](#5-visualizing-the-results)
  - [Finding Potential Frauds](#6-finding-potential-frauds)
  - [Printing the Fraud Clients](#7-printing-the-fraud-clients)

---

## Project Overview

This project uses a **Self-Organizing Map** to process a dataset of credit card applications and identify potentially fraudulent ones. The SOM clusters similar applications together and highlights outliers that might represent fraud. The project includes:

- Training a SOM using customer data.
- Visualizing the resulting SOM grid with color-coded distinctions between fraudulent and non-fraudulent applications.
- Listing the IDs of customers whose applications are identified as potentially fraudulent.

---

## Libraries and Dependencies

Before running the script, make sure to install the following Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn minisom

```
NumPy: A library for numerical computations.
Pandas: A data manipulation tool to handle CSV files.
Matplotlib: A plotting library for visualization.
scikit-learn: Used for feature scaling.
MiniSom: A lightweight implementation of Self-Organizing Maps.

## Dataset
The dataset used for this project is Credit_Card_Applications.csv, which contains credit card application data. The last column in the dataset represents whether the application was approved (1) or not (0).

Input Features (X): All columns except the last one (application details).
Output Label (y): The last column, which indicates approval (1) or rejection (0).

## Code Explanation
# Importing the libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
We begin by importing the necessary libraries:
- NumPy: Used for handling numerical data and array manipulations.
 - Pandas: Helps in loading and managing the dataset.
 - Matplotlib: For plotting and visualizing the results of the SOM.

#  Importing the dataset
```python
dataset = pd.read_csv('Self_Organizing_Maps/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```
We load the dataset of credit card applications:
 - `pd.read_csv()`: Loads the dataset from a CSV file.
 - `X`: Contains all columns except the last one (which holds the credit card application details).
 - `y`: Contains the last column of the dataset, representing whether the application was approved (1) or not (0).

#  Feature Scaling
```python
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)
```
Scaling the features is crucial for SOMs because they are sensitive to the magnitude of input data.
We use MinMaxScaler from scikit-learn to scale all features to a range between 0 and 1.
- `MinMaxScaler`: Scales features to a specified range (0 to 1 in this case).
- `fit_transform()`: Fits the scaler to the dataset and transforms it to the scaled range.

#  Training the SOM
```python
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)
```
 We train the Self-Organizing Map (SOM) using the MiniSom library:
 - `MiniSom(x, y, input_len)`: Initializes a 10x10 SOM grid, where input_len is the number of features (15).
 - `sigma=1.0`: Controls how much neighboring neurons influence each other.
 - `learning_rate=0.5`: Defines the rate at which the SOM adjusts its weights.
 - `random_weights_init(X)`: Initializes the SOM with random weights for each neuron.
 - `train_random()`: Trains the SOM on the dataset for 100 iterations, clustering data points into neurons.

#  Visualizing the results with separate index box
```python
from pylab import bone, pcolor, colorbar, plot, show
import matplotlib.patches as mpatches

# Initialize the plot
bone()
pcolor(som.distance_map().T)  # plot the distance map (U-Matrix)
colorbar()  # add color bar
```
 We visualize the U-Matrix (Unified Distance Matrix), which represents distances between neighboring neurons:
 - `bone()`: Initializes the plot.
 - `pcolor()`: Plots the U-Matrix, showing distances between neurons.
 - `colorbar()`: Adds a color bar to help interpret the map.
```python
# Define markers and colors for known fraud (y=1) and non-fraud (y=0)

markers = ['o', 's']
colors = ['r', 'g']

# Loop through each data point to mark on the SOM
for i, x in enumerate(X):
    w = som.winner(x)  # get the winning node for each data point
    plot(w[0] + 0.5, w[1] + 0.5, 
         markers[y[i]], 
         markeredgecolor=colors[y[i]], 
         markerfacecolor='None', 
         markersize=10, 
         markeredgewidth=2)
```
 Fraudulent applications are represented by red circles, while non-fraudulent ones are green squares.
 - `som.winner()`: Identifies the winning neuron for each data point.
 - `plot()`: Plots the data points on the SOM with specific markers and colors.

# Creating a separate legend box
```python
fraud_patch = mpatches.Patch(color='red', label='Fraudulent (Red Circles)')
non_fraud_patch = mpatches.Patch(color='green', label='Non-Fraudulent (Green Squares)')

# Adding the legend
plt.legend(handles=[fraud_patch, non_fraud_patch], loc='upper right', bbox_to_anchor=(1.1, 1), borderaxespad=0.)

# Display the plot
show()
```
 The legend box helps distinguish between fraudulent and non-fraudulent applications.
 - `mpatches.Patch()`: Defines the legend elements.
 - `plt.legend()`: Adds the legend at the specified position.
 - `show()`: Displays the final SOM plot.
```python
#  Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,2)]), axis=0)
frauds = sc.inverse_transform(frauds)
```
 After visualizing the SOM, we identify potentially fraudulent applications.
 - `win_map(X)`: Maps each data point to its neuron on the SOM.
 - `np.concatenate()`: Combines the data points mapped to neurons (8,1) and (6,2), which likely contain frauds.
 - `sc.inverse_transform()`: Inversely transforms the scaled fraud data back to its original form.
```python
#  Printing the Fraud Clients
print('Fraud Customer IDs')
for i in frauds[:, 0]:
    print(int(i))
```
 Finally, we print the customer IDs of applications flagged as fraudulent.
 - `frauds[:, 0]`: Retrieves the customer IDs of the suspected frauds.
 - `print()`: Outputs the fraud customer IDs.
