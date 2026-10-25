from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import time

X,y = fetch_openml('mnist_784', version=1,return_X_y = True, as_frame=False, parser='auto')
X = X[:10000]/255.0
y = y[:10000]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
activations = ['logistic', 'tanh','relu']
color = ['blue','green','red']
plt.figure(figsize = (8,5))

for act, col in zip(activations, color):
    model =  MLPClassifier(hidden_layer_sizes=(50,) , activation = act, max_iter = 2000, random_state = 42)

    start_time = time.time()
    model.fit(X_train,y_train)
    end_time = time.time()
    acc = model.score(X_test,y_test)*100
    t_time = end_time - start_time
    print("Activation", act.upper())
    print("Time Taken:", t_time)
    print("Accurasy:", acc)
    print("-"*30)
    plt.plot(model.loss_curve_,label=act.upper(),color=col)

plt.title("Convergence Graph")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()
