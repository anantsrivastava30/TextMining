
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[1]:

import pandas as pd
import re
import numpy 


# In[2]:

def date_sorter():

    doc = []
    with open('dates.txt') as file:
        for line in file:
            doc.append(line)

    df = pd.Series(doc)
    #isinstance(df, pd.Series)
        
    # Your code here
    isinstance(df, pd.Series)
    
    df = df.str.replace(r'(?<=[/,])\d\d(?=[ |.=\n)])', lambda x : '19' + x.group(0))
    # '-' can be used to indicate minute to minute relationship
    df = df.str.replace(r'(?<=[-]\d\d[-])\d\d(?=[ |.=\n)])', lambda x : '19' + x.group(0))
    #isinstance(df, pd.Series)
    
    data = df.str.extract(r"""
                (
                (?<![.])\d{1,2} |  Jan(?:[au][au]ry)? | Feb(?:ruary)? | Mar(?:ch)? | Apr(?:il)? | May? | Jun(?:e)? | Jul(?:y)? |
                           Aug(?:ust)? | Sep(?:tember)? | Oct(?:ober)? | Nov(?:ember)? | Dec(?:em(?:e)?ber)? 
                )??
                [\W]*?
                (
                \d{1,2} |  Jan(?:[au][au]ry)? | Feb(?:ruary)? | Mar(?:ch)? | Apr(?:il)? | May | Jun(?:e)? | Jul(?:y)? |
                           Aug(?:ust)? | Sep(?:tember)? | Oct(?:ober)? | Nov(?:ember)? | Dec(?:em(?:e)?ber)? 
                )?
                [\W]*?
                ((?<!\d\d\d[-])
                \d\d\d\d
                )
                """,re.X)

    null_col = data.columns[data.isnull().any()]
    #print(data[null_col].isnull().sum())
    data.update(data[data[1].isnull()].fillna('1'))
    data.update(data[data[0].isnull()].fillna('1'))

    # print(data.head(200).tail(30))
    #print(isinstance(df, pd.Series))
    #print(isinstance(data, pd.DataFrame))

    x = data.as_matrix()
    count = 0
    for row in range(x.shape[0]):
        count += 1 
        if not x[row][1].isdigit() and x[row][0].isdigit():
            x[row][0], x[row][1] = x[row][1], x[row][0]
        
    #print(count)
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

    data[0] = data.apply(lambda x : int(x[0]), axis=1)
    data[1] = data.apply(lambda x : int(x[1]), axis=1)
    data[2] = data.apply(lambda x : int(x[2]), axis=1)
    data.sort_index(inplace=True,sort_remaining=True,by=[2,0,1])
    # data = data.apply(lambda x : x if x[1].isdigit() else x.reindex([2,0,1]), axis=1)
    # print(data.index.values)
    
    return pd.Series(data.index.values)


# In[3]:

date_sorter()


# In[ ]:



