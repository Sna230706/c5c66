from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
iris = load_iris()
df = pd.DataFrame(iris.data, column =  iris.feature_names)
df['target'] = iris.target
X = iris.data[:,:2]
y = (iris.target == 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm_model = SVC (kernel='linear',
C=1.0,
random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
def plot_svm_boundary(svm_model, X, y):
    DecisionBoundaryDisplay.from_estimator(  
            svm_model, X, response_method="predict", cmap=plt.cm.coolwarm, alpha=0.3,
            xlabel='sepal length (scaled)',ylabel='sepal widtth (scaled)')
    plt.scatter(X[:,0],X[:,1],cmap = plt.cm.coolwarm, c=y, edgecolor="k")
    plt.scatter(svm_model.support_vectors_[:,0],svm_model.support_vectors_[:,1],
    s=100, linewidth=1, facecolors='none', edgecolor='k',label = 'Support Vectors')
    plt.title('Linear SVM Decision Boundary')
    plt.legend()
    plt.show()
plot_svm_boundary(svm_model, X_train_scaled, y_train)
