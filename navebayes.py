import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 1. Load Dataset

categories = ['sci.space', 'rec.sport.baseball', 'comp.graphics']

data = fetch_20newsgroups(subset='all',
                         categories=categories,
                         remove=('headers', 'footers', 'quotes'))

X = data.data
y = data.target


# 2. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. Feature Extraction


# Multinomial (word counts)
vectorizer_multi = CountVectorizer(stop_words='english', max_features=5000)
X_train_multi = vectorizer_multi.fit_transform(X_train)
X_test_multi = vectorizer_multi.transform(X_test)

# Bernoulli (binary features)
vectorizer_bern = CountVectorizer(stop_words='english', binary=True, max_features=5000)
X_train_bern = vectorizer_bern.fit_transform(X_train)
X_test_bern = vectorizer_bern.transform(X_test)


# 4. Train Models


# Multinomial NB
mnb = MultinomialNB()
mnb.fit(X_train_multi, y_train)

# Bernoulli NB
bnb = BernoulliNB()
bnb.fit(X_train_bern, y_train)


# 5. Predictions

y_pred_mnb = mnb.predict(X_test_multi)
y_pred_bnb = bnb.predict(X_test_bern)


# 6. Evaluation

print("=== Multinomial NB ===")
print("Accuracy:", accuracy_score(y_test, y_pred_mnb))
print("F1 Score:", f1_score(y_test, y_pred_mnb, average='macro'))
print(classification_report(y_test, y_pred_mnb))

print("\n=== Bernoulli NB ===")
print("Accuracy:", accuracy_score(y_test, y_pred_bnb))
print("F1 Score:", f1_score(y_test, y_pred_bnb, average='macro'))
print(classification_report(y_test, y_pred_bnb))