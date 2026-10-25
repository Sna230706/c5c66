<<<<<<< HEAD
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score

data = fetch_openml(name='adult',version=2)
df = data.frame

x = df.drop(columns=['class'])
x = pd.get_dummies(x)

y = df['class'].apply(lambda x: 1 if '>50K' in str(x) else 0)
x_train , x_test , y_train , y_test =train_test_split(x,y , test_size = 0.2 , random_state = 42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train_scaled,y_train)
lr_pred = lr_model.predict(x_test_scaled)
dt_model = DecisionTreeClassifier(max_depth=5)
dt_model.fit(x_train_scaled,y_train)
dt_pred = dt_model.predict(x_test_scaled)
lr_acc = accuracy_score(y_test,lr_pred)
dt_acc = accuracy_score(y_test,dt_pred)
lr_prec = precision_score(y_test,lr_pred)
dt_prec = precision_score(y_test,dt_pred)
lr_recall = recall_score(y_test,lr_pred)
dt_recall = recall_score(y_test,dt_pred)
lr_f1 = f1_score(y_test,lr_pred)
dt_f1 = f1_score(y_test,dt_pred)
