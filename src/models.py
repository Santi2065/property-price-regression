import numpy as np
import pandas as pd

from metrics import mse  
from data_splitting import train_val_split

# Regresión lineal (pseudoinversa/gradiente) con L2/L1
class RegresionLineal:
    """
    - pseudoinversa (Ridge si lambda_l2>0)
    - gradiente (admite L2 y L1)
    """
    def __init__(self, X, y, feature_names=None, lambda_l2=0.0, lambda_l1=0.0):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.n, self.d = self.X.shape
        self.feature_names = feature_names if feature_names is not None else ["Intercepto"] + [f"x{i}" for i in range(self.d)]
        self.lambda_l2 = float(lambda_l2)
        self.lambda_l1 = float(lambda_l1)
        self.coef_ = None

    def _add_bias(self, X):
        return np.c_[np.ones(X.shape[0]), X]

    def fit_pseudoinversa(self):
        Xb = self._add_bias(self.X)
        XtX = Xb.T @ Xb
        I = np.eye(XtX.shape[0])
        I[0, 0] = 0.0

        if self.lambda_l2 > 0.0:
            A = XtX + self.lambda_l2 * I
            self.coef_ = np.linalg.pinv(A) @ (Xb.T @ self.y)
        else:
            self.coef_ = np.linalg.pinv(Xb) @ self.y
        return self

    def fit_gradiente(self, lr=1e-2, n_iter=5000, tol=1e-6):
        Xb = self._add_bias(self.X)
        n, d1 = Xb.shape
        w = self.coef_.copy() if getattr(self, "coef_", None) is not None else np.zeros(d1)

        lam2, lam1 = self.lambda_l2, self.lambda_l1
        for _ in range(int(n_iter)):
            resid = Xb @ w - self.y
            grad = (2.0 / n) * (Xb.T @ resid)

            # L2 (ridge)
            if lam2 > 0.0:
                g_l2 = 2.0 * lam2 * w
                g_l2[0] = 0.0
                grad += g_l2

            # paso de gradiente 
            w_tmp = w - lr * grad

            # Proximal L1: soft-threshold en TODAS menos el intercepto
            if lam1 > 0.0:
                thr = lr * lam1
                w_new = w_tmp.copy()
                # intercepto no se regulariza
                w_new[0] = w_tmp[0]
                # soft-thresholding
                w_new[1:] = np.sign(w_tmp[1:]) * np.maximum(np.abs(w_tmp[1:]) - thr, 0.0)
            else:
                w_new = w_tmp

            # early stopping por cambio en pesos
            if np.linalg.norm(w_new - w, ord=2) < tol:
                w = w_new
                break

            w = w_new
    
        self.coef_ = w
        return self

    def predict(self, X_new):
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        Xb = self._add_bias(X_new)
        return Xb @ self.coef_

    def print_coefs(self):
        print("Coeficientes:")
        for name, c in zip(self.feature_names, self.coef_):
            print(f"{name}: {c:.6f}")

    def coef_of(self, feature_name: str) -> float:
        idx = self.feature_names.index(feature_name)
        return float(self.coef_[idx])


# Base interna para M1..M5
class _BaseModel:
    def __init__(self, method="pinv", lr=1e-2, n_iter=5000, lambda_l2=0.0, lambda_l1=0.0,
                 test_size=0.2, random_state=42):
        self.method = method
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.test_size = test_size
        self.random_state = random_state
        self.report_ = None
        self.model_ = None
        self.features_ = None

    def _alias_area(self, df):
        """Si viene 'area' sin tilde, crear 'Área'."""
        df = df.copy()
        if "Área" not in df.columns and "area" in df.columns:
            df["Área"] = df["area"]
        return df

    def _fit_core(self, df_train, df_val, features):
        X_tr = df_train[features].astype(float).to_numpy()
        y_tr = df_train["precio"].to_numpy()

        model = RegresionLineal(
            X_tr, y_tr,
            feature_names=["Intercepto"] + list(features),
            lambda_l2=self.lambda_l2,
            lambda_l1=self.lambda_l1
        )
        if self.method == "pinv":
            model.fit_pseudoinversa()
        else:
            model.fit_gradiente(lr=self.lr, n_iter=self.n_iter)

        yhat_tr = model.predict(X_tr)
        mse_tr = mse(y_tr, yhat_tr)

        X_va = df_val[features].astype(float).to_numpy()
        y_va = df_val["precio"].to_numpy()
        yhat_va = model.predict(X_va)
        mse_va = mse(y_va, yhat_va)

        self.model_ = model
        self.features_ = list(features)
        self.report_ = {
            "features": list(features),
            "method": self.method,
            "mse_train": float(mse_tr),
            "mse_val": float(mse_va),
            "coefs": model.coef_.copy()
        }
        return self

    def predict(self, dfX: pd.DataFrame):
        assert self.model_ is not None and self.features_ is not None, "Primero llamá a fit()."
        dfX = self._alias_area(dfX)
        for c in self.features_:
            if c not in dfX.columns:
                dfX[c] = 0.0
        return self.model_.predict(dfX[self.features_].astype(float).to_numpy())

    def print_coefs(self):
        assert self.model_ is not None
        self.model_.print_coefs()


# M1 / M2 / M3 (6 features)
class M1(_BaseModel):
    """precio ~ Área"""
    def fit(self, df_proc: pd.DataFrame):
        df = self._alias_area(df_proc)
        df_tr, df_va = train_val_split(df, test_size=self.test_size, random_state=self.random_state)
        return self._fit_core(df_tr, df_va, ["Área"])


class M2(_BaseModel):
    """precio ~ Área + pileta (pileta es bool)"""
    def fit(self, df_proc: pd.DataFrame):
        df = self._alias_area(df_proc)
        feats = ["Área", "pileta"]
        for c in feats:
            if c not in df.columns:
                df[c] = 0.0
        df_tr, df_va = train_val_split(df, test_size=self.test_size, random_state=self.random_state)
        return self._fit_core(df_tr, df_va, feats)

    def efecto_pileta_en_usd(self) -> float:
        """Coef marginal de tener pileta (en USD)."""
        assert self.model_ is not None
        return self.model_.coef_of("pileta")


class M3(_BaseModel):
    """precio ~ 6 características (consigna 3.3)"""
    def __init__(self, *args, features=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._default_feats = ["Área", "pileta", "edad", "ambientes", "pisos", "tipo_casa"]
        self.custom_features = features

    def fit(self, df_proc: pd.DataFrame):
        df = self._alias_area(df_proc)
        feats = self.custom_features or self._default_feats
        for c in feats:
            if c not in df.columns:
                df[c] = 0.0
        df_tr, df_va = train_val_split(df, test_size=self.test_size, random_state=self.random_state)
        return self._fit_core(df_tr, df_va, feats)


# M4 (M3 + derivadas) y M5 (~50 features)
class M4(_BaseModel):
    """precio ~ M3 + derivadas (polinomios grado 2 + algunas interacciones)"""
    def __init__(self, *args, base_features_M3=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_features = base_features_M3 or ["Área", "pileta", "edad", "ambientes", "pisos", "tipo_casa"]

    def _build_features(self, df):
        df = self._alias_area(df)
        for c in self.base_features:
            if c not in df.columns:
                df[c] = 0.0

        X = df[self.base_features].copy().astype(float)

        # polinomios grado 2 de numéricas
        num_base = [c for c in self.base_features if pd.api.types.is_numeric_dtype(df[c])]
        for c in num_base:
            X[f"{c}^2"] = (df[c].astype(float) ** 2)

        # algunas interacciones (limitadas)
        inter_candidates = [c for c in num_base][:4]
        for i in range(len(inter_candidates)):
            for j in range(i+1, len(inter_candidates)):
                c1, c2 = inter_candidates[i], inter_candidates[j]
                X[f"{c1}*{c2}"] = df[c1].astype(float) * df[c2].astype(float)

        return X

    def fit(self, df_proc: pd.DataFrame):
        X = self._build_features(df_proc)
        y = df_proc["precio"]
        df_all = pd.concat([X, y], axis=1)

        df_tr, df_va = train_val_split(df_all, test_size=self.test_size, random_state=self.random_state)


class M5(_BaseModel):
    """precio ~ ~50 features (polinomios e interacciones)"""
    def __init__(self, *args, base_features_M3=None, target_count=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_features = base_features_M3 or ["Área", "pileta", "edad", "ambientes", "pisos", "tipo_casa"]
        self.target_count = int(target_count)

    def _build_raw_features(self, df):
        df = self._alias_area(df)
        for c in self.base_features:
            if c not in df.columns:
                df[c] = 0.0

        X = df[self.base_features].astype(float).copy()

        # polinomios grado 2 y 3 de numéricas
        num_base = [c for c in self.base_features if pd.api.types.is_numeric_dtype(df[c]) and c != "pileta"]
        for c in num_base:
            X[f"{c}^2"] = (df[c].astype(float) ** 2)
            X[f"{c}^3"] = (df[c].astype(float) ** 3)

        # interacciones de todas las numéricas
        inter_candidates = [*num_base, "pileta"] if "pileta" in self.base_features else num_base
        for i in range(len(inter_candidates)):
            for j in range(i+1, len(inter_candidates)):
                c1, c2 = inter_candidates[i], inter_candidates[j]
                X[f"{c1}*{c2}"] = df[c1].astype(float) * df[c2].astype(float)

        # recortar a ~target_count si nos pasamos
        if X.shape[1] > self.target_count:
            X = X.iloc[:, :self.target_count]

        return X

    def _build_features(self, df):
        """Genera features y, si hay stats aprendidas, estandariza con mu/std del TRAIN."""
        X = self._build_raw_features(df)
        stats = getattr(self, "_mu_std_", None)
        if stats is not None:
            # aplicar estándar columna a columna
            for col in X.columns:
                mu, sd = stats.get(col, (0.0, 1.0))
                sd = sd if sd not in (0.0, None) else 1.0
                X[col] = (X[col] - mu) / sd
        return X

    def fit(self, df_proc: pd.DataFrame):
        # Genero features crudas, hago split y aprendo mu/std SOLO del TRAIN
        X_all_raw = self._build_raw_features(df_proc)
        y_all = df_proc["precio"]
        df_all = pd.concat([X_all_raw, y_all], axis=1)

        df_tr, df_va = train_val_split(df_all, test_size=self.test_size, random_state=self.random_state)

        # stats del TRAIN
        mu_std = {}
        for col in df_tr.columns:
            if col == "precio":
                continue
            mu = float(df_tr[col].mean())
            sd = float(df_tr[col].std() or 1.0)
            mu_std[col] = (mu, sd)
        self._mu_std_ = mu_std

        # aplicar estándar a train/val
        Xtr = df_tr.drop(columns=["precio"]).copy()
        Xva = df_va.drop(columns=["precio"]).copy()
        for col in Xtr.columns:
            mu, sd = mu_std[col]
            sd = sd if sd not in (0.0, None) else 1.0
            Xtr[col] = (Xtr[col] - mu) / sd
            Xva[col] = (Xva[col] - mu) / sd

        df_tr_std = pd.concat([Xtr, df_tr["precio"].reset_index(drop=True)], axis=1)
        df_va_std = pd.concat([Xva, df_va["precio"].reset_index(drop=True)], axis=1)

        return self._fit_core(df_tr_std, df_va_std, list(Xtr.columns))
