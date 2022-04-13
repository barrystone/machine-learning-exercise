import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 0, strategy = 'mean')
imputer.fit(x[:,0:3])
x[:,0:3] = imputer.transform(x[:,0:3])

imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imputer.fit(x[:,[3]])
x[:,[3]] = imputer.transform(x[:,[3]])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('State', OneHotEncoder(),[3])],remainder = 'passthrough')
x = ct.fit_transform(x)

x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state =1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.api as sm
x_train = np.append(arr = np.ones((40, 1)).astype(int), values = x_train, axis = 1)
X_opt = x_train[:, [0, 1, 2, 3, 4, 5]]
#X_opt = np.array(X_opt, dtype=float)
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =  y_train , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = x_train[:, [0, 1, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =  y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = x_train[:, [0, 3, 4, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =  y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = x_train[:, [0, 3, 5]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =  y_train, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = x_train[:, [0, 3]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =  y_train, exog = X_opt).fit()
regressor_OLS.summary()

 