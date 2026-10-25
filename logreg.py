import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("Pima_indian_diabetes.csv")
X = df.drop("Outcome",axis = 1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
without_scaling = LogisticRegression(
max_iter=1000,
random_state=42
)
without_scaling.fit(X_train, y_train)
pred_without = without_scaling.predict(X_test)
accuracy_without = accuracy_score(y_test, pred_without)
Precision_without = precision_score(y_test, pred_without)
recall_without = recall_score(y_test, pred_without)
f1_without = f1_score(y_test, pred_without)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
with_scaling = LogisticRegression(
max_iter=1000,
random_state=42
)
with_scaling.fit(X_train_scaled, y_train)
pred_with = with_scaling.predict(X_test_scaled)
accuracy_with = accuracy_score(y_test, pred_with)
precision_with = precision_score(y_test, pred_with)
recall_with = recall_score(y_test, pred_with)
f1_score = f1_score(y_test, pred_with)
