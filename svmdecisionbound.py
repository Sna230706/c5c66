import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# Load dataset
iris = datasets.load_iris()
X = iris.data[:, :2]   # take first 2 features
y = iris.target

# Convert to binary classification (Setosa vs Non-Setosa)
y = np.where(y == 0, 1, -1)

# Train Linear SVM
model = svm.SVC(kernel='linear', C=1)
model.fit(X, y)

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y)

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

Z = model.decision_function(xy).reshape(XX.shape)

# Decision boundary
ax.contour(XX, YY, Z, levels=[0])

# Margins
ax.contour(XX, YY, Z, levels=[-1, 1], linestyles=['--', '--'])

# Support vectors
ax.scatter(model.support_vectors_[:, 0],
           model.support_vectors_[:, 1],
           s=100, facecolors='none', edgecolors='k')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Linear SVM on Iris Dataset")
plt.show()