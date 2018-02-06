import pandas as pd
import numpy as np

df = pd.read_csv("Amazon_Unlocked_Mobile.csv")

df = df.sample(frac=0.01, random_state=10)
df.dropna(inplace=True)
df = df[df['Rating'] != 3]

df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)

print(df['Positively Rated'].mean())
