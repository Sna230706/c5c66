import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# 2. Load Dataset

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 3. Preprocessing

# Flatten images (28x28 → 784)
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0


# 4. Experiment with different K

k_values = [1, 3, 5]

for k in k_values:
    print(f"\n===== K = {k} =====")
    
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train[:10000], y_train[:10000])  # subset for speed
    
    y_pred = knn.predict(X_test[:2000])
    
    acc = accuracy_score(y_test[:2000], y_pred)
    
    print("Accuracy:", acc)
    print(classification_report(y_test[:2000], y_pred))