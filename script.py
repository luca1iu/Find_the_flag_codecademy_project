import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name', 'landmass', 'zone', 'area', 'population', 'language', 'religion', 'bars', 'stripes', 'colours',
        'red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'mainhue', 'circles',
        'crosses', 'saltires', 'quarters', 'sunstars', 'crescent', 'triangle', 'icon', 'animate', 'text', 'topleft',
        'botright']
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names=cols)

# variable names to use as predictors
var = ['red', 'green', 'blue', 'gold', 'white', 'black', 'orange', 'mainhue', 'bars', 'stripes', 'circles', 'crosses',
       'saltires', 'quarters', 'sunstars', 'triangle', 'animate']

# Print number of countries by landmass, or continent
print(df['landmass'].value_counts())

# Create a new dataframe with only flags from Europe and Oceania
df_36 = df[df.landmass.isin([3, 6])]

# Print the average vales of the predictors for Europe and Oceania
print(df_36.groupby('landmass')[var].mean())

# Create labels for only Europe and Oceania
labels = (df['landmass'].isin([3, 6])) * 1

# Print the variable types for the predictors
print(df.dtypes)

# Create dummy variables for categorical predictors
data = pd.get_dummies(df[var])
print(data)
# Split data into a train and test set
x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=0, test_size=0.4)

# Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth)
    tree.fit(x_train, y_train)
    acc_depth.append(tree.score(x_test, y_test))

# Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.show()
# Find the largest accuracy and the depth this occurs
print(np.max(acc_depth))
print(acc_depth.index(np.max(acc_depth)))

# Refit decision tree model with the highest accuracy and plot the decision tree
tree1 = DecisionTreeClassifier(max_depth=depths[1])
tree1.fit(x_train, y_train)

plt.figure(figsize=(27, 12))
tree.plot_tree(tree1)
plt.tight_layout()
plt.show()

# Create a new list for the accuracy values of a pruned decision tree.  Loop through
# the values of ccp and append the scores to the list



# Plot the accuracy vs ccp_alpha


# Find the largest accuracy and the ccp value this occurs


# Fit a decision tree model with the values for max_depth and ccp_alpha found above


# Plot the final decision tree
