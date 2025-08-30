import pandas as pd
import numpy as np

def handle_missing_values(df):
    df = df.copy()
    df = df[df.get("precio", 0) > 1]
    if "edad" in df.columns and "tipo" in df.columns:
        df["edad"] = df.groupby("tipo")["edad"].transform(lambda s: s.fillna(s.median()))
    if "pisos" in df.columns:
        df["pisos_missing"] = df["pisos"].isna().astype(int)
        if "tipo" in df.columns:
            df.loc[df["tipo"] != "casa", "pisos"] = 1
            med_casa = df.loc[df["tipo"] == "casa", "pisos"].median()
            df.loc[(df["tipo"] == "casa") & (df["pisos"].isna()), "pisos"] = med_casa
        df["pisos"] = df["pisos"].fillna(1)
    df = df.drop(columns=["unidades"], errors="ignore")
    return df.fillna(0)

def one_hot_encode(df, column_name):
    if column_name not in df.columns:
        return df
    dummies = pd.get_dummies(df[column_name], prefix=column_name, dtype=int)
    df = df.drop(columns=[column_name])
    df = pd.concat([df, dummies], axis=1)
    return df

def normalize(df, column_name, stats=None):
    df = df.copy()
    x = df[column_name].astype(float)
    if stats is None:
        mean = float(x.mean()); std = float(x.std() or 1.0)
        stats = {"method": "zscore", "mean": mean, "std": std}
    mean = stats["mean"]; std = stats["std"] or 1.0
    df[column_name] = (x - mean) / std
    return df, stats

