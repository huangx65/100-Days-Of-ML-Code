import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

DATA_FILE_PATH = "../datasets/Data.csv"


dataset = pd.read_csv(DATA_FILE_PATH)
print(dataset)
X = dataset.iloc[:, 0:3].values
Y = dataset.iloc[:, -1].values
print(X, "\n", Y)
print(type(X), type(Y))

# Replace missing values by mean value
imputer = SimpleImputer(
    missing_values=np.nan, strategy="mean"
)  # np.nan and mean are default
X[:, 1:] = imputer.fit_transform(X=X[:, 1:])
print(X)

# Encoding categorical data, make string inputs transform to numerical values
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
print(X)

# Using one-hot label to avoid weight for label
ct = ColumnTransformer(
    [("one-hot encoder", OneHotEncoder(), [0])], remainder="passthrough"
)
X = ct.fit_transform(X)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(X, Y)

# Splitting the datasets into training sets and Test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print(X_train, X_test, Y_train, Y_test)
