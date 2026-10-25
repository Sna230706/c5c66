from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,f1_score
target_categories = [
 'comp.graphics',
 'sci.med',
 'talk.politics.guns',
 'alt.atheism'
]
newspapper = fetch_20newsgroups(
 subset='all',
 categories=target_categories,
 random_state=42,
 remove=('headers','footer','quotes')
)
x = newspapper.data
y = newspapper.target
x_train , x_test, y_train , y_test = train_test_split(
 x,y,test_size=0.25,random_state = 42,stratify=y
)
tfidf_vectorizer = TfidfVectorizer(stop_words='english',max_features=5000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)
countvectorizer = CountVectorizer(binary=True,stop_words='english',max_features=5000)
x_train_bin = countvectorizer.fit_transform(x_train)
x_test_bin = countvectorizer.transform(x_test)
mnb_model = MultinomialNB()
mnb_model.fit(x_train_tfidf,y_train)
mnb_pred = mnb_model.predict(x_test_tfidf)
bnb_model = BernoulliNB()
bnb_model.fit(x_train_bin,y_train)
bnb_pred = bnb_model.predict(x_test_bin)
mnb_acc = accuracy_score(y_test,mnb_pred)
mnb_f1 = f1_score(y_test,mnb_pred,average='weighted')
bnb_acc = accuracy_score(y_test,bnb_pred)
bnb_f1 = f1_score(y_test,bnb_pred,average='weighted')