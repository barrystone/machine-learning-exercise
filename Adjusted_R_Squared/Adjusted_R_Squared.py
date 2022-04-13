import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('insurance_simple.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 0, strategy = 'mean')
imputer.fit(x[:,0:2])
x[:,0:2] = imputer.transform(x[:,0:2])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2, random_state =1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.api as sm
x_train = np.append(arr = np.ones((1070, 1)).astype(int), values = x_train, axis = 1)

X_opt = x_train[:, [0, 1, 2, 3]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =  y_train , exog = X_opt).fit()
regressor_OLS.summary()

X_opt = x_train[:, [0, 1, 2]]
X_opt = X_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog =  y_train , exog = X_opt).fit()
regressor_OLS.summary()
