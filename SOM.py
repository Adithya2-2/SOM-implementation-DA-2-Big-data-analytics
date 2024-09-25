"""### Importing the libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""## Importing the dataset"""

dataset = pd.read_csv('Self_Organizing_Maps\Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

"""## Feature Scaling"""

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

"""##Training the SOM"""

from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


"""## Visualizing the results with separate index box"""

from pylab import bone, pcolor, colorbar, plot, show
import matplotlib.patches as mpatches

# Initialize the plot
bone()
pcolor(som.distance_map().T)  # plot the distance map (U-Matrix)
colorbar()  # add color bar

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

# Creating a separate legend box
fraud_patch = mpatches.Patch(color='red', label='Fraudulent (Red Circles)')
non_fraud_patch = mpatches.Patch(color='green', label='Non-Fraudulent (Green Squares)')

# Adding the legend
plt.legend(handles=[fraud_patch, non_fraud_patch], loc='upper right', bbox_to_anchor=(1.1, 1), borderaxespad=0.)

# Display the plot
show()



"""## Finding the frauds"""

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,2)]), axis = 0)
frauds = sc.inverse_transform(frauds)

"""##Printing the Fraud Clients"""

print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))
