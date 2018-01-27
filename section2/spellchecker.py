from nltk.corpus import words
import pandas as pd

# from ..section1.dates import date_sorter

correct_spellings = words.words()
spelling_series = pd.Series(correct_spellings)


def jaccard(entries, grams):
    """
    Args:
      entries : the words to be corrected
      grams : the distribution of letters

    Return:
      list : of corrected words
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
        outcomes.append(closest)

    return outcomes
