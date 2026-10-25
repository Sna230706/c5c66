import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
df = pd.read_csv("winequality-red.csv",sep =";")
X = df.drop('quality', axis=1)
y = df['quality']
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
architectures = {
 "shallow&narrow" : (8,),
 "shallow&wide" : (128,),
 "moderate depth" : (64,32),
 "deep&high capacity" : (128,64,32)
}
for name , arch in architectures.items():
 mlp = MLPClassifier(hidden_layer_sizes=arch, activation='relu',solver='adam',max_iter=1000, random_state =42)
 mlp.fit(X_train_scaled,y_train)
 y_pred = mlp.predict(X_test_scaled)
 accuracy = accuracy_score(y_test,y_pred)
 print(f"Architecture: {name} {arch}")
 print(f"Accuracy: {accuracy:.4f}")
 print("-"*40)
