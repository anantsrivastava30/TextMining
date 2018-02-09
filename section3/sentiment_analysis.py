# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re


df = pd.read_csv("Amazon_Unlocked_Mobile.csv")

df = df.sample(frac=0.01, random_state=10)
df.dropna(inplace=True)
df = df[df['Rating'] != 3]

df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)

print(df['Positively Rated'].mean())

X_train, X_test, y_train, y_test = train_test_split(
    df['Reviews'], df['Positively Rated'], random_state=0)
print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)

# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
print(vect.get_feature_names()[::2000])
print(len(vect.get_feature_names()))
X_train_vectorized = vect.transform(X_train)
print(X_train_vectorized.toarray().shape)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target'] == 'spam', 1, 0)
spam_data.head(10)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    spam_data['text'], spam_data['target'], random_state=0)


def answer_one():
    print(spam_data.shape)

    return "{:2.3f}%".format(spam_data['target'].values.mean() * 100)


from sklearn.feature_extraction.text import CountVectorizer


def answer_two():
    vect = CountVectorizer().fit(X_train)
    return sorted(vect.get_feature_names(), key=len)[-1]


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score


def answer_three():
    vect = CountVectorizer().fit(X_train)
    x_trained_vector = vect.transform(X_train)
    print(x_trained_vector.shape)
    model = MultinomialNB()
    model.fit(x_trained_vector, y_train)
    predictions = model.predict(vect.transform(X_train))

    return roc_auc_score(y_train, predictions)


def answer_four():
    vect = TfidfVectorizer().fit(X_train)
    s1 = pd.Series(vect.idf_, vect.get_feature_names())
    s2 = pd.Series(vect.idf_, vect.get_feature_names())
    s1.sort_values(inplace=True, ascending=True)
    s2.sort_values(inplace=True, ascending=False)

    return (s1[:20], s2[:20])


def answer_five():
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    print(X_train_vectorized.shape)
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))

    return roc_auc_score(y_test, predictions)


def answer_six():
    sp = spam_data[spam_data['target'] == 1]
    _ns = sp['text'].apply(lambda x: len(x)).mean()
    sp = spam_data[spam_data['target'] == 0]
    _s = sp['text'].apply(lambda x: len(x)).mean()
    return (_ns, _s)


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


def answer_seven():
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    print(X_train_vectorized.shape)

    sd = spam_data['text'].apply(lambda x: len(x))

    X_train_vectorized_new = add_feature(
        X_train_vectorized, sd[X_train.index])
    model = SVC(C=10000)
    model.fit(X_train_vectorized_new, y_train)
    X_test_new = add_feature(vect.transform(X_test), sd[X_test.index])
    predictions = model.predict(X_test_new)

    return roc_auc_score(y_test, predictions)


def answer_eight():
    spam = spam_data[spam_data['target'] == 1]
    n_spam = spam_data[spam_data['target'] == 0]
    digit = spam['text'].apply(lambda s: sum(
        [c.isdigit() for c in s])).mean()
    n_digit = n_spam['text'].apply(
        lambda s: sum([c.isdigit() for c in s])).mean()
    return (digit, n_digit)


def answer_nine():
    vect = TfidfVectorizer(min_df=5, ngram_range=(1, 3)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    print(X_train_vectorized.shape)
    sd = spam_data['text'].apply(lambda x: len(x))
    dc = spam_data['text'].apply(
        lambda s: sum([c.isdigit() for c in s]))
    X_train_vectorized_new = add_feature(
        X_train_vectorized, sd[X_train.index])
    X_train_vectorized_new = add_feature(
        X_train_vectorized, dc[X_train.index])

    model = LogisticRegression(
        C=100).fit(
        X_train_vectorized_new, y_train)
    X_test_new = add_feature(vect.transform(X_test), sd[X_test.index])
    X_test_new = add_feature(vect.transform(X_test), dc[X_test.index])
    prediction = model.predict(X_test_new)
    return roc_auc_score(y_test, prediction)


def answer_ten():
    ss = spam_data['text'].apply(
        lambda s: sum([bool(re.match('\W', x)) for x in s]))
    spam = ss[spam_data['target'] == 1].mean()
    n_spam = ss[spam_data['target'] == 0].mean()

    return (spam, n_spam)


def test_sample(v, model):
    se = pd.Series(["A lot has happened on Facebook since , XIME offers a 2 year, full-time, residential PGDM programme through its centers last logged in. Here are some notifications you've missed from your friends",
                    "As you might be aware, XIME offers a 2 year, full-time, residential PGDM programme through its centers in Bangalore, Chennai and Kochi. Our PGDM programme is ranked 22nd by Business India (All India - 2017) and 25th by Business Standard (All India - 2017)."])
    ld = se.apply(lambda s: len(s))
    nd = se.apply(lambda s: sum([c.isdigit() for c in s]))
    nw = se.apply(lambda s: sum([bool(re.match('\W', x)) for x in s]))
    X = add_feature(v.transform(se), [ld, nd, nw])
    print("sample case predictions :{}".format(model.predict(X)))


def answer_eleven():
    vect = CountVectorizer(
        min_df=5, ngram_range=(2, 5), analyzer='char_wb').fit(X_train)

    ld = spam_data['text'].apply(lambda s: len(s))
    nd = spam_data['text'].apply(
        lambda s: sum([c.isdigit() for c in s]))
    nw = spam_data['text'].apply(
        lambda s: sum([bool(re.match('\W', x)) for x in s]))

    X_train_vectorized = add_feature(
        vect.transform(X_train),
        [ld[X_train.index], nd[X_train.index], nw[X_train.index]])

    model = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    X_test_vectorized = add_feature(
        vect.transform(X_test),
        [ld[X_test.index], nd[X_test.index], nw[X_test.index]])
    predection = model.predict(X_test_vectorized)
    features = np.array(vect.get_feature_names())
    coff = model.coef_[0].argsort()
    large_coff = features[coff[:10]]
    small_coeff = features[coff[len(coff) - 11:len(coff) - 1]]
    roc = roc_auc_score(y_test, predection)
    return (roc, list(small_coeff), list(large_coff))
