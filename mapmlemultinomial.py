from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , confusion_matrix ,classification_report
data = fetch_20newsgroups(
 subset = 'all',
 remove = ('headers','footers','quotes')
)
target_categories = ['alt.atheism','sci.electronics','comp.graphics','sci.med']
newsgroup = fetch_20newsgroups(
 subset = 'all',
 categories= target_categories,
 remove = ('headers','footers','quotes')
)
x = newsgroup.data
y = newsgroup.target
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size = 0.2,random_state =42)
vectorizer = CountVectorizer()
vect_x_train = vectorizer.fit_transform(x_train)
vect_x_test = vectorizer.transform(x_test)
mle_model = MultinomialNB(alpha = 1e-10)
mle_model.fit(vect_x_train , y_train)
mle_pred = mle_model.predict(vect_x_test)
mle_acc = accuracy_score(y_test ,mle_pred)
priors = [0.01 , 0.5 , 1.0 ,1.5]
for alpha in priors :
 map_model = MultinomialNB(alpha = alpha)
 map_model.fit(vect_x_train,y_train)
 map_pred = map_model.predict(vect_x_test)
 map_accuracy = accuracy_score(y_test, map_pred)
 print(f"MAP Accuracy (Prior in alpha ={alpha} :{map_accuracy:.4f}")

best_model_map = MultinomialNB(alpha = 0.01)
best_model_map.fit(vect_x_train , y_train)
best_model_pred = best_model_map.predict(vect_x_test)
print(classification_report(y_test,best_model_pred))
