import numpy as np

def train_val_split(df, test_size=0.2, random_state=None):
    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_size = int(round((1 - test_size) * len(df)))
    df_train = shuffled.iloc[:train_size]
    df_val   = shuffled.iloc[train_size:]
    return df_train, df_val

def cross_val():
    return