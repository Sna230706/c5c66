#import library
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#load datasets
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)
df["Target"] = diabetes.target

#extract features and target
x = df.drop('Target', axis = 1)
y = df['Target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

 #preprocessing
scaler =  StandardScaler()
x_train_scaler =  scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

#Linear Regression
lr_model =  LinearRegression()
lr_model.fit(x_train_scaler, y_train)
lr_model_pred = lr_model.predict(x_test_scaler)

lr_mse =  mean_squared_error(y_test, lr_model_pred)
lr_r2 = r2_score(y_test, lr_model_pred)

#alphas
Alphas =np.logspace(-3,3,50)

ridge_grid =  GridSearchCV(
    estimator = Ridge(),
    param_grid = {'alpha' : Alphas},
    scoring = 'neg_mean_squared_error',
    cv = 5
)

ridge_grid.fit(x_train_scaler, y_train)
best_ridge = ridge_grid.best_estimator_

ridge_pred = ridge_grid.predict(x_test_scaler)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_r2 =  r2_score(y_test, ridge_pred)

#Lasso
lasso_grid =  GridSearchCV(
    estimator = Lasso(max_iter = 10000),
    param_grid = {'alpha' : Alphas},
    scoring = 'neg_mean_squared_error',
    cv = 5
)

lasso_grid.fit(x_train_scaler, y_train)
best_lasso = lasso_grid.best_estimator_

lasso_pred = lasso_grid.predict(x_test_scaler)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_r2 = r2_score(y_test, lasso_pred)

#visualization
model = ['Linear Regression', 'Ridge', 'Lasso']
mse_values = [lr_mse, ridge_mse, lasso_mse]
r2_score = [lr_r2, ridge_r2, lasso_r2]

fig, axes =  plt.subplots(1,2, figsize = (10,6))

axes[0].bar(model, mse_values, color = ['Blue','Red', 'Black'])
axes[0]. set_title("Mean_Squared_error")
axes[0].set_ylabel('mse')

axes[1].bar(model, r2_score, color = ['Blue','Red', 'Black'])
axes[1]. set_title("R2_Score")
axes[1].set_ylabel('r2_score')
plt.show()