import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# == Importing the dataset ==
dataset = pd.read_csv('insurance.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# == Taking care of missing data ==
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:,0:1])
x[:,0:1] = imputer.transform(x[:,0:1])
imputer.fit(x[:,2:4])
x[:,2:4] = imputer.transform(x[:,2:4])


# == Encoding categorical data ==
## Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[1,4,5])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
## Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# == Splitting the dataset into the Training set and Test set ==
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state =1)

# == Feature Scaling (Standardisation method) ==
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,8:] = sc.fit_transform(x_train[:,8:])
x_test[:,8:] = sc.transform(x_test[:,8:])


## Showing results
print("\n == x ==")
print(x)
print("\n == y ==")
print(y)
print("\n==============")

print("\n == x_train ==")
print(x_train)
print("\n == x_test ==")
print(x_test)
print("\n == y_train ==")
print(y_train)
print("\n == y_test ==")
print(y_test)
print("\n\n\n")

# == Training the Multiple Linear Regression model on the Training set ==
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print("== Predicting the Test set results (Multiple Linear Regression) ==")
print(y_pred)

print("\n == Predicting single result (row-50) ==")
print(regressor.predict([[0.0,1.0,1.0,0.0,0.0,0.0,1.0,0.0,1.5752125501075516,0.1469699840269174,-0.9070577122378711
]]))
print("row-50: ", y_pred[50])

