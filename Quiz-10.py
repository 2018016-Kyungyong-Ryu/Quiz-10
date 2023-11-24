import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split

data_coluumns = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
data = pd.read_csv("./data/09_irisdata.csv", names = data_coluumns)

print(np.shape(data))
print(data.describe())
print(data.groupby('class').size())
scatter_matrix(data)
plt.savefig("./data/scatter_plot.png")

X = data.iloc[:, 0:4].values
Y = data.iloc[:,4].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)


kfold = KFold(n_splits=10, random_state=5, shuffle=True)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
result = cross_val_score(model, X, Y, cv=kfold)

print(result.mean())

