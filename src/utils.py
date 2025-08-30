import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def agregar_ciudad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ciudad" not in df.columns and "lat" in df.columns and "lon" in df.columns:
        df['ciudad'] = 'Desconocida'
        df.loc[(df['lat'] > 30) & (df['lat'] < 45) & (df['lon'] < -70) & (df['lon'] > -80), 'ciudad'] = 'Nueva York'
        df.loc[(df['lat'] < -30) & (df['lat'] > -40) & (df['lon'] < -55) & (df['lon'] > -65), 'ciudad'] = 'Buenos Aires'
    if "nyc" not in df.columns and "ciudad" in df.columns:
        df["nyc"] = (df["ciudad"] == "Nueva York").astype(int)
    return df

def kfold_indices(n: int, k: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield tr_idx, val_idx

def cv_linear_reg(X, y, lambdas, kind="ridge", k=5, lr=1e-2, n_iter=15000, feature_names=None):
    from models import RegresionLineal
    from metrics import mse
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).reshape(-1)
    means, stds = [], []
    for lam in lambdas:
        fold_mses = []
        for tr_idx, va_idx in kfold_indices(X.shape[0], k=k, seed=42):
            Xt, yt = X[tr_idx], y[tr_idx]
            Xv, yv = X[va_idx], y[va_idx]
            if kind == "ridge":
                model = RegresionLineal(Xt, yt, feature_names=feature_names, lambda_l2=float(lam))
                model.fit_pseudoinversa()
            elif kind == "lasso":
                model = RegresionLineal(Xt, yt, feature_names=feature_names, lambda_l1=float(lam))
                model.fit_gradiente(lr=lr, n_iter=n_iter)
            else:
                raise ValueError("kind debe ser 'ridge' o 'lasso'")
            fold_mses.append(mse(yv, model.predict(Xv)))
        means.append(np.mean(fold_mses)); stds.append(np.std(fold_mses))
    return np.array(means), np.array(stds)

def learning_curve(build_Xy_fn, tag, df_tr, df_va, sizes_frac, rng_seed=42, **build_kwargs):
    from metrics import rmse
    Xv, yv, _ = build_Xy_fn(tag, df_va, **build_kwargs)
    n = len(df_tr)
    sizes_n = np.unique([max(20, int(fr*n)) for fr in sizes_frac])
    rmse_tr, rmse_va = [], []
    rng = np.random.default_rng(rng_seed)
    idx = np.arange(n); rng.shuffle(idx)
    for m in sizes_n:
        take = idx[:m]
        Xtr, ytr, f_names_tr = build_Xy_fn(tag, df_tr.iloc[take], **build_kwargs)
        from models import RegresionLineal
        model = RegresionLineal(Xtr, ytr, feature_names=f_names_tr)
        model.fit_pseudoinversa()
        rmse_tr.append(rmse(ytr, model.predict(Xtr)))
        rmse_va.append(rmse(yv, model.predict(Xv)))
    return sizes_n, np.array(rmse_tr), np.array(rmse_va)

def plot_learning_curve(sizes_n, rmse_tr, rmse_va, title="Learning Curve"):
    plt.figure(figsize=(6,4))
    plt.plot(sizes_n, rmse_tr, marker='o', label='Train')
    plt.plot(sizes_n, rmse_va, marker='o', label='Val')
    plt.xlabel("N muestras"); plt.ylabel("RMSE"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()
