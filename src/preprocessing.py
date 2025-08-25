import pandas as pd
import numpy as np

def handle_missing_values(df):
    df = df.copy()  # no pisar el original

    # 1. Quitar filas sin target (precio)
    df = df[~df["precio"].isna()]

    # 2. Edad → imputar con mediana por tipo de propiedad
    df["edad"] = df.groupby("tipo")["edad"].transform(
        lambda s: s.fillna(s.median())
    )

    # 3. Pisos → manejar NaNs
    #    a) Crear flag para marcar los NaN originales
    df["pisos_missing"] = df["pisos"].isna().astype(int)

    #    b) Para deptos/PH, setear 0 porque no aplica
    df.loc[df["tipo"] != "casa", "pisos"] = 0

    #    c) Para casas, imputar con la mediana de casas
    med_casa = df.loc[df["tipo"] == "casa", "pisos"].median()
    df.loc[(df["tipo"] == "casa") & (df["pisos"].isna()), "pisos"] = med_casa

    # 4. borrar la columna unidades
    df = df.drop(columns=["unidades"], errors="ignore")

    # 5. rellenar otros NaNs residuales con 0
    df = df.fillna(0)

    return df

# Función para aplicar One-Hot Encoding a una columna
def one_hot_encode(df, column_name):
    one_hot = pd.get_dummies(df[column_name], prefix=column_name)
    df = df.drop(column_name, axis=1)
    df = df.join(one_hot)
    return df

# Función para normalizar una columna numérica utilizando z-score normalization
def normalize(df, column_name):
    df[column_name] = (df[column_name] - df[column_name].mean()) / df[column_name].std()
    return df

def train_test_split(df, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    shuffled = df.sample(frac=1)
    train_size = int((1 - test_size) * len(df))
    df_train = shuffled[:train_size]
    df_val = shuffled[train_size:]
    return df_train, df_val