import pandas as pd
import re
import numpy as np


def date_sorter():
    """
    This function takes a record
    and extracts the dates
    :arg:
        :Input file
    :return:
        :List of dates in sorted order
    """

    doc = []
    with open('dates.txt') as file:
        for line in file:
            doc.append(line)

    df = pd.Series(doc)
    # isinstance(df, pd.Series)

    # Your code here
    isinstance(df, pd.Series)

    df = df.str.replace(r'(?<=[/,])\d\d(?=[ |.=\n)])', lambda x: '19' + x.group(0))
    # '-' can be used to indicate minute to minute relationship
    df = df.str.replace(r'(?<=[-]\d\d[-])\d\d(?=[ |.=\n)])', lambda x: '19' + x.group(0))
    # isinstance(df, pd.Series)

    # print(df[490])
    # print(df[490])
    data = df.str.extract(r"""
                (
                \d{1,2} |  Jan(?:[au][au]ry)? | Feb(?:ruary)? | Mar(?:ch)? | Apr(?:il)? | May? | Jun(?:e)? | Jul(?:y)? |
                           Aug(?:ust)? | Sep(?:tember)? | Oct(?:ober)? | Nov(?:ember)? | Dec(?:em(?:e)?ber)? 
                )??
                [\W]*?
                (
                (?<![\d])\d{1,2} |  Jan(?:[au][au]ry)? | Feb(?:ruary)? | Mar(?:ch)? | Apr(?:il)? | May | Jun(?:e)? | Jul(?:y)? |
                           Aug(?:ust)? | Sep(?:tember)? | Oct(?:ober)? | Nov(?:ember)? | Dec(?:em(?:e)?ber)? 
                )?
                [\W]*?
                ((?<!\d\d\d[-])
                \d\d\d\d
                )
                """, re.X)

    # print(data.values)

    null_idx = (data[0].isnull()) & (data[1].notnull())
    data.loc[null_idx, 0] = data[1]
    data.loc[null_idx, 1] = np.nan

    # print(data[null_idx])

    null_col = data.columns[data.isnull().any()]
    print(data[null_col].isnull().sum())
    data.update(data[data[1].isnull()].fillna('1'))
    data.update(data[data[0].isnull()].fillna('1'))

    x = data.as_matrix()
    count = 0
    for row in range(x.shape[0]):
        count += 1
        if x[row][0].isdigit() and not x[row][1].isdigit():
            x[row][0], x[row][1] = x[row][1], x[row][0]

    # print(data.head(200).tail(30))
    # print(isinstance(df, pd.Series))
    # print(isinstance(data, pd.DataFrame))
    np.savetxt('data.txt', data.values, fmt="%s %s %s")
    # print(count)
    data.replace(r'Ja.*', '1', inplace=True, regex=True)
    data.replace(r'Feb.*', '2', inplace=True, regex=True)
    data.replace(r'Mar.*', '3', inplace=True, regex=True)
    data.replace(r'Apr.*', '4', inplace=True, regex=True)
    data.replace(r'May.*', '5', inplace=True, regex=True)
    data.replace(r'Jun.*', '6', inplace=True, regex=True)
    data.replace(r'Jul.*', '7', inplace=True, regex=True)
    data.replace(r'Aug.*', '8', inplace=True, regex=True)
    data.replace(r'Sep.*', '9', inplace=True, regex=True)
    data.replace(r'Oct.*', '10', inplace=True, regex=True)
    data.replace(r'Nov.*', '11', inplace=True, regex=True)
    data.replace(r'Dec.*', '12', inplace=True, regex=True)

    data[0] = data.apply(lambda x: int(x[0]), axis=1)
    data[1] = data.apply(lambda x: int(x[1]), axis=1)
    data[2] = data.apply(lambda x: int(x[2]), axis=1)

    great_idx = (data[0] > 12)
    data.loc[great_idx, 0] = 1

    data.sort_index(inplace=True, sort_remaining=True, by=[2, 0, 1])
    np.savetxt('data_sorted.txt', data.values, fmt="%d %d %d")
    # data = data.apply(lambda x : x if x[1].isdigit() else x.reindex([2,0,1]), axis=1)
    # print(data.index.values)
    return pd.Series(data.index.values)


