from nltk.corpus import words
import pandas as pd
import numpy as np

# from ..section1.dates import date_sorter

correct_spellings = words.words()
spelling_series = pd.Series(correct_spellings)


def jaccard(entries, grams):
    """
    :parameter
      :param entries:the words to be corrected
      :param grams:the distribution of letters

    Return:
      :return list of corrected words
    """
    N = grams
    outcomes = []
    for entry in entries:
        spellings = spelling_series[spelling_series.str.startswith(entry[0])]
        set_one = [entry[i:i + N] for i in range(len(entry) - N + 1)]
        dist = []
        for word in spellings:
            set_two = [word[i:i + N] for i in range(len(word) - N + 1)]
            jacc = len(set(set_one).intersection(set_two)) / len(set(set_one).union(set_two))
            dist.append((jacc, word))

        closest = max(dist)
        outcomes.append(closest[1])

    return outcomes


def dl_transposition(entry, word):
    """
    calculates the Damerau-Levenshtien distance

    :parameter:
      :param word:
      :param entry: : list of misspelled words

    Returns:
      :return distance
    """
    #     print(type(entry),type(word))
    da = np.zeros(shape=(26), dtype=float)
    d = np.zeros((len(entry) + 2, len(word) + 2))
    maxdist = len(entry) + len(word)
    d[0, 0] = maxdist
    for i in range(1, len(entry) + 2):
        d[i, 0] = maxdist
        d[i, 1] = i
    for j in range(1, len(word) + 2):
        d[0, j] = maxdist
        d[1, j] = j
    for i in range(2, len(entry) + 2):
        db = 0
        for j in range(2, len(word) + 2):
            k = da[ord(word[j - 2]) - 97]
            l = db
            if entry[i - 2] == word[j - 2]:
                cost = 0
                db = j
            else:
                cost = 1
            # print(i,j,l,k)
            d[i, j] = min(d[i - 1, j - 1] + cost, d[i, j - 1] + 1, d[i - 1, j] + 1, d[k - 1, l - 1] + 1 + (j - l + 1))
        da[ord(entry[i - 2]) - 97] = i
    return d[len(entry), len(word)]


def dl_distance(entries):
    outcomes = []
    for entry in entries:
        spellings = spelling_series[spelling_series.str.startswith(entry[0])]
        dis = []
        for spelling in spellings:
            dis.append((dl_transposition(entry, spelling), spelling))
        d = min(dis)
        outcomes.append(d[1])
    return outcomes

