import numpy as np
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load dataset
from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Reduce size (for faster training in lab)
X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_test[:1000]
y_test = y_test[:1000]

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Flatten images
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# -------- Linear SVM --------
linear_model = svm.SVC(kernel='linear')
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# -------- Polynomial SVM --------
poly_model = svm.SVC(kernel='poly', degree=3)
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

# -------- RBF SVM --------
rbf_model = svm.SVC(kernel='rbf')
rbf_model.fit(X_train, y_train)
y_pred_rbf = rbf_model.predict(X_test)

# Accuracy
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("Polynomial SVM Accuracy:", accuracy_score(y_test, y_pred_poly))
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))