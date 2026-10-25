from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = load_breast_cancer()
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
df["Diagnosis"] = dataset.target

x = df.drop('Diagnosis', axis = 1)
y = df['Diagnosis']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

scaler =  StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler =  scaler.transform(x_test)

mle_model =  LogisticRegression(
    penalty= None,
    max_iter = 10000,
)
mle_model.fit(x_train_scaler, y_train)

map_l2_model = LogisticRegression(
    penalty= 'l2',
    C = 0.1,
    max_iter = 10000,
)

map_l2_model.fit(x_train_scaler, y_train)

map_l1_model = LogisticRegression(
    penalty= 'l1',
    solver = 'liblinear',
    C = 0.1
)

map_l1_model.fit(x_train_scaler, y_train)

mle_pred = mle_model.predict(x_test_scaler)
mle_proba = mle_model.predict_proba(x_test_scaler)

map_l2_pred = map_l2_model.predict(x_test_scaler)
map_l2_proba = map_l2_model.predict_proba(x_test_scaler)

map_l1_pred = map_l1_model.predict(x_test_scaler)
map_l1_proba = map_l1_model.predict_proba(x_test_scaler)

accuracy_mle = accuracy_score(y_test, mle_pred)
Log_loss = log_loss(y_test, mle_proba)

accuracy_l2 = accuracy_score(y_test, map_l2_pred)
Log_loss_l2 = log_loss(y_test, map_l2_proba)

accuracy_l1 = accuracy_score(y_test, map_l1_pred)
Log_loss_l1 = log_loss(y_test, map_l1_proba)

plt.figure(figsize= (10,6))

features = dataset.feature_names
x_indices = np.arange(len(features))
plt.plot(x_indices, mle_model.coef_[0], marker = 'o', label = 'MLE no reg')
plt.plot(x_indices, map_l2_model.coef_[0], marker = 's', label = 'MAP L2')
plt.plot(x_indices, map_l1_model.coef_[0], marker = '^', label = 'MAP L1')

plt.axhline(0, color = 'gray', linewidth = 1)
plt.xticks(x_indices, features, rotation = 90)
plt.xlabel("Features")
plt.ylabel("Weight values")
plt.title("Comparisons estimation Parameters")
plt.legend()
plt.tight_layout()
plt.show()