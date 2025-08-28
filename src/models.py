import numpy as np
import numpy as np
import pandas as pd
from metrics import mse
from data_splitting import train_val_split

# -------------------------
# Helpers internos del módulo
# -------------------------
def _ensure_pileta_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un DF con columna 'pileta_1' binaria (0/1). No normalizar."""
    out = df.copy()
    
    # Already have pileta_1 column
    if "pileta_1" in out.columns:
        # Ensure it's truly binary
        out["pileta_1"] = out["pileta_1"].astype(float).astype(int)
        return out
        
    # Need to convert from pileta column
    if "pileta" in out.columns:
        # Handle different possible formats
        if out["pileta"].dtype == bool:
            # Boolean values
            out["pileta_1"] = out["pileta"].astype(int)
        elif pd.api.types.is_string_dtype(out["pileta"]):
            # String values like 'True'/'False', 'Yes'/'No', etc.
            truthy = ['true', 'yes', 'si', '1', 'y', 's', 't']
            out["pileta_1"] = out["pileta"].str.lower().isin(truthy).astype(int)
        else:
            # Try to convert numeric or other types
            out["pileta_1"] = out["pileta"].astype(float).fillna(0).astype(int)
        
        # Verify the conversion worked
        unique_vals = out["pileta_1"].unique()
        if not all(val in [0, 1] for val in unique_vals):
            print(f"Warning: pileta_1 contains non-binary values: {unique_vals}")
            # Force to binary as last resort
            out["pileta_1"] = (out["pileta_1"] > 0).astype(int)
            
        return out
        
    raise KeyError("No se encontró 'pileta' ni 'pileta_1' en el DataFrame.")

def _fit_and_report(df_train: pd.DataFrame, df_val: pd.DataFrame, features,
                    method="pinv", lr=1e-2, n_iter=5000):
    """Entrena RegresionLineal con las features dadas y devuelve métricas y el modelo."""
    X_tr = df_train[features]
    y_tr = df_train["precio"].to_numpy()

    model = RegresionLineal(X_tr, y_tr)
    if method == "pinv":
        model.fit_pseudoinversa()
    else:
        model.fit_gradiente(lr=lr, n_iter=n_iter)

    yhat_tr = model.predict(X_tr)
    mse_tr = mse(y_tr, yhat_tr)

    X_va = df_val[features]
    y_va = df_val["precio"].to_numpy()
    yhat_va = model.predict(X_va)
    mse_va = mse(y_va, yhat_va)

    return {
        "features": list(features),
        "method": method,
        "mse_train": float(mse_tr),
        "mse_val": float(mse_va),
        "model": model,
        "yhat_train": yhat_tr,
        "yhat_val": yhat_va,
    }

class RegresionLineal:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        #guardar nombres de features
        self.feature_names = ["Intercepto"] + list(X.columns)
        #inicializar coeficientes
        self.coef = None

    def fit_pseudoinversa(self):
        X_bias = np.c_[np.ones(self.X.shape[0]), self.X]
        self.coef = np.linalg.pinv(X_bias) @ self.y

    def fit_gradiente(self, lr=0.01, n_iter=1000):
        X_bias = np.c_[np.ones(self.X.shape[0]), self.X]
        n, d = X_bias.shape
        w = np.zeros(d)

        for _ in range(n_iter):
            grad = (2/n) * X_bias.T @ (X_bias @ w - self.y)
            w -= lr * grad

        self.coef = w
    
    def predict(self, X_new):
        X_new = np.array(X_new)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        X_bias = np.c_[np.ones(X_new.shape[0]), X_new]
        return X_bias @ self.coef
    
    def print_coefs(self, feature_names):
        print("Coeficientes:")
        for name, coef in zip(feature_names, self.coef):
            print(f"{name}: {coef:.4f}")

# -------------------------
# Clase M1: precio ~ Área
# -------------------------
class M1:
    def __init__(self, test_size=0.2, random_state=42, method="pinv",
                 lr=1e-2, n_iter=5000):
        self.features = ["Área"]
        self.test_size = test_size
        self.random_state = random_state
        self.method = method
        self.lr = lr
        self.n_iter = n_iter

        self.report_ = None
        self.model_ = None

    def fit(self, df_proc: pd.DataFrame):
        df_tr, df_va = train_val_split(df_proc, test_size=self.test_size,
                                       random_state=self.random_state)
        rep = _fit_and_report(df_tr, df_va, self.features, self.method, self.lr, self.n_iter)
        self.report_ = rep
        self.model_ = rep["model"]
        return self

    def predict(self, dfX: pd.DataFrame):
        assert self.model_ is not None, "Primero llamá a fit()."
        return self.model_.predict(dfX[self.features])

    def print_coefs(self):
        assert self.model_ is not None, "Primero llamá a fit()."
        self.model_.print_coefs(["Intercepto"] + self.features)

# ---------------------------------------
# Clase M2: precio ~ Área + pileta (0/1)
# ---------------------------------------
class M2:
    def __init__(self, test_size=0.2, random_state=42, method="pinv",
                 lr=1e-2, n_iter=5000):
        self.base_features = ["Área"]
        self.features = ["Área", "pileta_1"]
        self.test_size = test_size
        self.random_state = random_state
        self.method = method
        self.lr = lr
        self.n_iter = n_iter

        self.report_ = None
        self.model_ = None

    def fit(self, df_proc: pd.DataFrame):
        df_proc = _ensure_pileta_binary(df_proc)
        df_tr, df_va = train_val_split(df_proc, test_size=self.test_size,
                                       random_state=self.random_state)
        rep = _fit_and_report(df_tr, df_va, self.features, self.method, self.lr, self.n_iter)
        self.report_ = rep
        self.model_ = rep["model"]
        return self

    def predict(self, dfX: pd.DataFrame):
        assert self.model_ is not None, "Primero llamá a fit()."
        dfX = _ensure_pileta_binary(dfX)
        return self.model_.predict(dfX[self.features])

    def print_coefs(self):
        assert self.model_ is not None, "Primero llamá a fit()."
        self.model_.print_coefs(["Intercepto"] + self.features)

    # interpretación del efecto de pileta
    def efecto_pileta_en_usd(self):
        """Devuelve el coeficiente de pileta_1 (delta promedio en USD, ceteris paribus)."""
        assert self.model_ is not None, "Primero llamá a fit()."
        # coef: [w0, w_area, w_pileta]
        return float(self.model_.coef[2])