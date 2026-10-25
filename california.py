from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns =  housing.feature_names)
df["MedHouseVal"] = housing.target

x = df["AveRooms"].values.reshape(-1,1)
y =  df["MedHouseVal"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state  = 42)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

gd_model = SGDRegressor(max_iter = 1000, learning_rate = "constant", eta0 = 0.01, random_state = 42)
gd_model.fit(x_train_scaler, y_train)
gd_model_pred = gd_model.predict(x_test_scaler)

normal_model = LinearRegression()
normal_model.fit(x_train_scaler, y_train)
normal_model_pred = normal_model.predict(x_test_scaler)

gd_mse = mean_squared_error(y_test, gd_model_pred)
gd_r2 = r2_score(y_test, gd_model_pred)

normal_mse = mean_squared_error(y_test, normal_model_pred)
normal_r2 = r2_score(y_test, normal_model_pred)


idx = np.argsort(x_test_scaler[:, 0])

plt.scatter(x_test_scaler, y_test, color='blue', label='Actual Data')
plt.plot(x_test_scaler[idx], gd_model_pred[idx], 'r-', label='Gradient Descent')
plt.plot(x_test_scaler[idx], normal_model_pred[idx], 'g-', label='Normal Equation')

plt.xlabel("Average Rooms")
plt.ylabel("House Value")
plt.title("GD vs Normal Equation")
plt.legend()
plt.show()