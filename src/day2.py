# Simple Linear Regession

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_FILE_PATH = "../datasets/studentscores.csv"


dataset = pd.read_csv(DATA_FILE_PATH)
print(dataset)
X = dataset.iloc[:, :1].values
Y = dataset.iloc[:, -1:].values
print(X, "\n", Y)
print(type(X), type(Y))


# Splitting the datasets into training sets and Test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_train, X_test, Y_train, Y_test)

# Fit linear regression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# predict test result
Y_pred = regressor.predict(X_test)

# plot training and testing dataset
plt.title("Training & Testing data")
plt.scatter(X_train, Y_train, color="red", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Fit line")

plt.scatter(X_test, Y_test, color="orange", label="Testing data")
plt.plot(X_test, regressor.predict(X_test), color="violet", label="Predict Result")
plt.legend()
plt.show()
