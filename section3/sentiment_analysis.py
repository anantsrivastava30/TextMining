import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.read_csv("Amazon_Unlocked_Mobile.csv")

df = df.sample(frac=0.01, random_state=10)
df.dropna(inplace=True)
df = df[df['Rating'] != 3]

df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)

print(df['Positively Rated'].mean())

X_train, X_test, y_train, y_test = train_test_split(df['Reviews'],
                                                    df['Positively Rated'],
                                                    random_state=0)
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
analyse
# Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))