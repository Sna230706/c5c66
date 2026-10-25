
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\sreen\OneDrive\Desktop\ml\auto-mpg.csv")
df = df.dropna()

# Feature and target
x = df["displacement"].values.reshape(-1, 1)
y = df["mpg"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Preprocessing
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

# Polynomial features (degree 2)
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train_scaler)
x_test_poly = poly.transform(x_test_scaler)

# Polynomial regression
poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
poly_pred = poly_model.predict(x_test_poly)

poly_mse = mean_squared_error(y_test, poly_pred)
poly_r2 = r2_score(y_test, poly_pred)

# Linear regression
lin_model = LinearRegression()
lin_model.fit(x_train_scaler, y_train)
lin_pred = lin_model.predict(x_test_scaler)

lin_mse = mean_squared_error(y_test, lin_pred)
lin_r2 = r2_score(y_test, lin_pred)

# Display results
print("Polynomial Regression")
print("MSE:", poly_mse)
print("R2:", poly_r2)

print("\nLinear Regression")
print("MSE:", lin_mse)
print("R2:", lin_r2)

# Visualisation
plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color="black", label="Actual Data")

x_grid = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
x_grid_scaler = scaler.transform(x_grid)
x_grid_poly = poly.transform(x_grid_scaler)

plt.plot(x_grid, lin_model.predict(x_grid_scaler),
         color="red", linewidth=2, label="Linear Fit")
plt.plot(x_grid, poly_model.predict(x_grid_poly),
         color="blue", linewidth=2, label="Polynomial Degree 2")

plt.title("MPG vs Displacement")
plt.xlabel("Displacement")
plt.ylabel("MPG")
plt.legend()
plt.show()
