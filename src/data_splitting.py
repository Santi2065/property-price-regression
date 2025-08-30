import numpy as np


def train_val_split(df, test_size=0.2, random_state=None):
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(df)
    n_val = int(round(n * float(test_size)))
    n_tr = n - n_val
    train_idx = np.arange(0, n_tr)
    val_idx = np.arange(n_tr, n)
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

def cross_val(X, y, lambdas, kind="ridge", k=5, lr=1e-2, n_iter=15000, feature_names=None, seed=42):
    from models import RegresionLineal
    from metrics import mse

    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).reshape(-1)
    idx = np.arange(len(X)); rng.shuffle(idx)
    folds = np.array_split(idx, k)

    means, stds = [], []
    for lam in lambdas:
        fold_mses = []
        for i in range(k):
            va_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            Xt, yt = X[tr_idx], y[tr_idx]
            Xv, yv = X[va_idx], y[va_idx]
            if kind == "ridge":
                mdl = RegresionLineal(Xt, yt, feature_names=feature_names, lambda_l2=float(lam))
                mdl.fit_pseudoinversa()
            else:
                mdl = RegresionLineal(Xt, yt, feature_names=feature_names, lambda_l1=float(lam))
                mdl.fit_gradiente(lr=lr, n_iter=n_iter)
            yhat = mdl.predict(Xv)
            fold_mses.append(mse(yv, yhat))
        means.append(float(np.mean(fold_mses)))
        stds.append(float(np.std(fold_mses)))
    return np.array(means), np.array(stds)