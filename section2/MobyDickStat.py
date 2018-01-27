import nltk
from nltk import WordNetLemmatizer
from nltk import FreqDist
import pandas as pd
import numpy as np
from collections import defaultdict
import operator
from nltk.corpus import words
from .spellchecker import jaccard

with open('moby.txt', 'r') as f:
    moby_raw = f.read()

moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)

print("There are {} tokens and there are {} "
      "unique tokens in Moby Dick".format(len(moby_tokens),
                                          len(set(moby_tokens))))  # or alternatively len(text1)

lemmatizer = WordNetLemmatizer()
lemmatized = pd.Series([lemmatizer.lemmatize(w, 'v') for w in text1])

print("After lemmatizing the verbs there are {} "
      "unique tokens in text!".format(len(set(lemmatized))))

print('The lexical diversity of the text is {:0.2f}'.format(len(set(moby_tokens)) /
                                                            len(moby_tokens)))

whales = (text1.count("Whale") + text1.count("whale")) / len(text1) * 100
print("{:0.2f}% of tokens are whales!".format(whales))

dist = FreqDist(text1)
df = pd.DataFrame([x for x in zip(dist.keys(), dist.values())])
df.sort_values(by=1, ascending=False, inplace=True)
df = df.head(20)
print("20 most frequently ocourring tokens in text :")
print(df.to_string(index=False))

print("\n Tokens below have length more than 5 and a frequency of more than 150!:")
print(sorted([w for w in dist.keys() if len(w) > 5 and dist[w] > 150]))


print('\n{!s} is the longest word in the text.'.format(str.title(sorted(text1, key=len)[-1])))

segement = [w for w in zip(dist.values(), dist.keys()) if w[0] > 2000 and w[1].isalpha()]
print("\nUnique words have a frequency of more than 2000:")
print(sorted(segement, key=lambda s: s[0], reverse=True))

print("There are an average of {:2.2f} tokens per sentence.".format(np.mean([len(nltk.word_tokenize(sent)) for
                                                                        sent in nltk.sent_tokenize(moby_raw)])))

fq = defaultdict(int)
for w in [x[1] for x in nltk.pos_tag(moby_tokens)]:
    fq[w] += 1

print("\n5 most frequent part of speect in this text:")
print(sorted(fq.items(), key=operator.itemgetter(1), reverse=True)[:5])

entries = ['cormulent', 'incendenece', 'validrate']