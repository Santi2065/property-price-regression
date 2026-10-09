"""Regenerates the figures and numbers shown in the README.

Re-runs the notebook's experiments with the code in src/ (the same models,
splits, grids and seeds as notebooks/Entrega_TP1.ipynb).

    pip install numpy pandas matplotlib
    python docs/figures/make_figures.py      # about two minutes (Lasso CV)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for f in Path("/usr/share/fonts/lm").glob("lm*10-*.otf"):  # Latin Modern, if installed
    font_manager.fontManager.addfont(str(f))
plt.style.use(HERE / "paper.mplstyle")
C = plt.rcParams["axes.prop_cycle"].by_key()["color"]

sys.path.insert(0, str(ROOT / "src"))
from metrics import mae, mse, rmse  # noqa: E402
from models import M1, M2, M3, M5, RegresionLineal  # noqa: E402
from utils import agregar_ciudad  # noqa: E402

CITY = {"Buenos Aires": C[0], "Nueva York": C[1]}
CITY_EN = {"Buenos Aires": "Buenos Aires", "Nueva York": "New York"}
SQFT = 0.09290304  # m^2 per square foot


def save(fig, name):
    fig.savefig(HERE / name, metadata={"Date": None})
    plt.close(fig)


def kusd(ax, axis="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:g}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---- data
raw = agregar_ciudad(pd.read_csv(ROOT / "data/raw/casas_dev.csv"))
raw = raw[raw["precio"] >= 2000].copy()  # same rows as data/processed/casas_dev.csv
raw["area_m2"] = np.where(raw["unidades"] == "sqft", raw["Área"] * SQFT, raw["Área"])
dev = agregar_ciudad(pd.read_csv(ROOT / "data/processed/casas_dev.csv"))
df_tr = agregar_ciudad(pd.read_csv(ROOT / "data/processed/casas_train.csv"))
df_va = agregar_ciudad(pd.read_csv(ROOT / "data/processed/casas_val.csv"))
df_te = agregar_ciudad(pd.read_csv(ROOT / "data/processed/casas_test.csv"))
FEATS6 = ["Área", "pileta", "edad", "ambientes", "pisos", "nyc"]

# ---- stages 2-3: M1-M3 on the 80/20 split of dev used by the model classes
m1 = M1(random_state=42).fit(dev)
m2 = M2(random_state=42).fit(dev)
m3 = M3(random_state=42, features=FEATS6).fit(dev)
stage3 = {k: np.sqrt(m.report_["mse_val"]) for k, m in (("M1", m1), ("M2", m2), ("M3", m3))}

# ---- stage 5: Ridge / Lasso on the M5 design (31 features, with the city indicator)
m5 = M5(base_features_M3=FEATS6, target_count=50, random_state=42).fit(df_tr)
cols = m5.features_
X_tr = m5._build_features(df_tr).reindex(columns=cols, fill_value=0.0).astype(float).to_numpy()
X_va = m5._build_features(df_va).reindex(columns=cols, fill_value=0.0).astype(float).to_numpy()
X_te = m5._build_features(df_te).reindex(columns=cols, fill_value=0.0).astype(float).to_numpy()
y_tr, y_va, y_te = (d["precio"].to_numpy() for d in (df_tr, df_va, df_te))
LASSO = dict(lr=5e-3, n_iter=60000, tol=1e-7)


def lam_max(X, y):
    return 2.0 / len(y) * np.max(np.abs((X - X.mean(0)).T @ (y - y.mean())))


def kfold(n, k=5, seed=42):
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        yield np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]


def cv(X, y, lambdas, kind):  # 5-fold CV, per-fold standardization, warm start for L1
    out = []
    for lam in lambdas:
        errs, prev = [], None
        for tr, va in kfold(len(y)):
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            Xt, Xv = (X[tr] - mu) / sd, (X[va] - mu) / sd
            if kind == "ridge":
                mdl = RegresionLineal(Xt, y[tr], lambda_l2=lam).fit_pseudoinversa()
            else:
                mdl = RegresionLineal(Xt, y[tr], lambda_l1=lam)
                if prev is not None:
                    mdl.coef_ = prev.copy()
                mdl.fit_gradiente(**LASSO)
                prev = mdl.coef_.copy()
            errs.append(mse(y[va], mdl.predict(Xv)))
        out.append((np.mean(errs), np.std(errs)))
    return np.array(out)


# coefficient paths on the training split
lam_l2 = np.logspace(-4, 4, 25)
path_l2 = np.array([RegresionLineal(X_tr, y_tr, lambda_l2=lam).fit_pseudoinversa().coef_ for lam in lam_l2])
lam_l1 = np.geomspace(lam_max(X_tr, y_tr), lam_max(X_tr, y_tr) * 1e-3, 25)
path_l1, prev = [], None
for lam in lam_l1:
    mdl = RegresionLineal(X_tr, y_tr, lambda_l1=lam)
    if prev is not None:
        mdl.coef_ = prev.copy()
    prev = mdl.fit_gradiente(**LASSO).coef_.copy()
    path_l1.append(prev)
path_l1 = np.array(path_l1)

grid_l2 = np.logspace(-3, 3, 20)
grid_l1 = lam_l1
cv_l2, cv_l1 = cv(X_tr, y_tr, grid_l2, "ridge"), cv(X_tr, y_tr, grid_l1, "lasso")
lam_r, lam_l = grid_l2[cv_l2[:, 0].argmin()], grid_l1[cv_l1[:, 0].argmin()]
m6r = RegresionLineal(X_tr, y_tr, lambda_l2=lam_r).fit_pseudoinversa()
m6l = RegresionLineal(X_tr, y_tr, lambda_l1=lam_l).fit_gradiente(**LASSO)

# ---- stage 6: final selection as submitted (designs built without the city indicator)
m5d = M5(random_state=42).fit(df_tr.drop(columns=["ciudad", "nyc"]))


def design(tag, df):
    if tag == "M1":
        return df[["Área"]].astype(float).to_numpy()
    if tag == "M2":
        return df[["Área", "pileta"]].astype(float).to_numpy()
    return m5d._build_features(df).reindex(columns=m5d.features_, fill_value=0.0).astype(float).to_numpy()


stage6 = {}
for tag in ("M1", "M2", "M5", "M6"):
    if tag == "M6":
        mdl = RegresionLineal(design(tag, df_tr), y_tr, lambda_l1=lam_l).fit_gradiente(**LASSO)
    else:
        mdl = RegresionLineal(design(tag, df_tr), y_tr).fit_pseudoinversa()
    stage6[tag] = rmse(y_va, mdl.predict(design(tag, df_va)))
# M4 cannot be refitted with the current src/models.py (M4.fit has no body after the split);
# its two numbers are the saved outputs of notebooks/Entrega_TP1.ipynb.
stage3["M4"], stage6["M4"] = np.sqrt(2184028063.681719), 70258.97

# test: M1 retrained on the whole dev set (submitted), and a post-hoc check of M6-Lasso
m1_dev = RegresionLineal(dev[["Área"]].astype(float).to_numpy(), dev["precio"].to_numpy()).fit_pseudoinversa()
yhat_m1 = m1_dev.predict(df_te[["Área"]].astype(float).to_numpy())
yhat_m6 = m6l.predict(X_te)

# ---- Figure 1: the two markets
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
for city, c in CITY.items():
    d = raw[raw["ciudad"] == city]
    ax.scatter(d["area_m2"], d["precio"], s=5, color=c, alpha=0.55, lw=0, label=f"{CITY_EN[city]} ($n={len(d)}$)")
ax.set_xlabel("area [m$^2$]")
ax.set_ylabel("price [kUSD]")
kusd(ax)
ax.set_title("(a) Price vs. area")
ax.legend(loc="upper right", markerscale=2.5)
ax = axes[1]
bins = np.geomspace(2e3, 1e6, 36)
for city, c in CITY.items():
    ax.hist(raw.loc[raw["ciudad"] == city, "precio"], bins=bins, color=c, alpha=0.6, label=CITY_EN[city])
ax.set_xscale("log")
ax.set_xlabel("price [USD]")
ax.set_ylabel("listings")
ax.set_title("(b) Price distribution")
fig.tight_layout()
save(fig, "fig1-markets.svg")

# ---- Figure 2: regularization paths and cross-validation
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
LABELS = {"Área": "area", "pileta": "pool", "edad": "age", "ambientes": "rooms", "pisos": "floors", "nyc": "nyc"}


def pretty(name):
    for es, en in LABELS.items():
        name = name.replace(es, en)
    return name.replace("*", r"$\times$").replace("^2", "$^2$").replace("^3", "$^3$")


# nyc is binary, so nyc^2 and nyc^3 are copies of it: plot each group of identical columns once
groups = {}
for j in range(len(cols)):
    groups.setdefault(X_tr[:, j].round(9).tobytes(), []).append(j)
groups = list(groups.values())
gname = {g[0]: ", ".join(pretty(cols[j]) for j in g) for g in groups}
top = []
for path, ref in ((path_l2, 0), (path_l1, -1)):
    order = sorted(groups, key=lambda g: -abs(path[ref, 1 + g[0]]))
    top += [g[0] for g in order[:4] if g[0] not in top]
for ax, lam, path, title in ((axes[0], lam_l2, path_l2, "(a) Ridge paths"), (axes[1], lam_l1, path_l1, "(b) Lasso paths")):
    for g in groups:
        if g[0] not in top:
            ax.semilogx(lam, path[:, 1 + g[0]] / 1e3, color="#c8c8c8", lw=0.6)
    for j, c in zip(top, C):
        ax.semilogx(lam, path[:, 1 + j] / 1e3, color=c, lw=1.1, label=gname[j])
    ax.axhline(0, color="#1a1a1a", lw=0.5)
    ax.set_xlabel(r"$\lambda$")
    ax.set_title(title)
axes[0].set_ylabel("coefficient [kUSD]")
ax = axes[2]
for g, res, lam, c, lab in ((grid_l2, cv_l2, lam_r, C[0], "Ridge"), (grid_l1, cv_l1, lam_l, C[1], "Lasso")):
    r = np.sqrt(res[:, 0]) / 1e3
    ax.semilogx(g, r, color=c, marker="o", ms=2.2, label=lab)
    ax.axvline(lam, color=c, ls=":", lw=0.8)
ax.set_xlabel(r"$\lambda$")
ax.set_ylabel("5-fold CV RMSE [kUSD]")
ax.set_title("(c) Cross-validation")
ax.legend(loc="upper left")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(top), bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(w_pad=0.6, rect=(0, 0.07, 1, 1))
save(fig, "fig2-regularization.svg")

# ---- Figure 3: validation RMSE per model, in the two stages of the notebook
models = ["M1", "M2", "M3", "M4", "M5", "M6"]
s3 = [stage3.get("M1"), stage3.get("M2"), stage3["M3"], stage3["M4"], np.nan, rmse(y_va, m6l.predict(X_va))]
s6 = [stage6["M1"], stage6["M2"], np.nan, stage6["M4"], stage6["M5"], stage6["M6"]]
fig, ax = plt.subplots(figsize=(7.2, 2.5))
x = np.arange(len(models))
ax.bar(x - 0.19, np.array(s3) / 1e3, 0.36, color=C[0], label="exploration (Sections 2–5)")
ax.bar(x + 0.19, np.array(s6) / 1e3, 0.36, color="#9e9e9e", label="final selection (Section 6.2)")
for xi, v in zip(x, s3):
    if np.isfinite(v):
        ax.text(xi - 0.19, v / 1e3 + 1, f"{v / 1e3:.1f}", ha="center", fontsize=7)
for xi, v in zip(x, s6):
    if np.isfinite(v):
        ax.text(xi + 0.19, v / 1e3 + 1, f"{v / 1e3:.1f}", ha="center", fontsize=7)
ax.text(4 - 0.19, 2, "NaN", ha="center", fontsize=7, color=C[0])
ax.set_xticks(x, ["M1\narea", "M2\n+pool", "M3\n6 feat.", "M4\n18 feat.", "M5\n31 feat.", "M6\nM5 + L1"])
ax.tick_params(axis="x", top=False, length=0)
ax.set_ylabel("validation RMSE [kUSD]")
ax.set_ylim(0, 97)
ax.legend(loc="upper center", ncol=2)
save(fig, "fig3-validation.svg")

# ---- Figure 4: predicted vs. actual price on the test set
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True, sharey=True)
lim = (min(0, yhat_m1.min(), yhat_m6.min()) - 2.5e4, 6.5e5)
for ax, yhat, title in ((axes[0], yhat_m1, "(a) M1 (selected), refit on dev"),
                        (axes[1], yhat_m6, "(b) M6 Lasso with city indicator")):
    for city, c in CITY.items():
        mask = (df_te["ciudad"] == city).to_numpy()
        ax.scatter(y_te[mask], yhat[mask], s=6, color=c, alpha=0.6, lw=0, label=CITY_EN[city])
    ax.plot(lim, lim, color="#1a1a1a", lw=0.7, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_title(title)
    ax.set_xlabel("actual price [kUSD]")
    ax.text(0.04, 0.95, f"RMSE {rmse(y_te, yhat) / 1e3:.1f}k\nMAE {mae(y_te, yhat) / 1e3:.1f}k",
            transform=ax.transAxes, va="top", fontsize=8)
    kusd(ax, "x")
    kusd(ax)
axes[0].set_ylabel("predicted price [kUSD]")
axes[1].legend(loc="lower right", markerscale=2.5)
fig.tight_layout()
save(fig, "fig4-test.svg")

# ---- numbers for the README tables
print("raw rows kept:", len(raw), raw.groupby("ciudad").size().to_dict(),
      "median price:", raw.groupby("ciudad")["precio"].median().round(0).to_dict())
print("M1 coefs", m1.report_["coefs"], "| M3 coefs", dict(zip(["b"] + FEATS6, m3.report_["coefs"].round(0))))
print("stage 2-5 val RMSE:", {k: round(v, 2) for k, v in stage3.items()})
print("M5 (dev split) val MSE:", M5(base_features_M3=FEATS6, random_state=42).fit(dev).report_["mse_val"])
print(f"Ridge lambda*={lam_r:.5f} CV MSE={cv_l2[:, 0].min():.4e} ± {cv_l2[cv_l2[:, 0].argmin(), 1]:.4e}"
      f" | train RMSE={rmse(y_tr, m6r.predict(X_tr)):.2f} val RMSE={rmse(y_va, m6r.predict(X_va)):.2f}")
print(f"Lasso lambda*={lam_l:.5f} CV MSE={cv_l1[:, 0].min():.4e} ± {cv_l1[cv_l1[:, 0].argmin(), 1]:.4e}"
      f" | train RMSE={rmse(y_tr, m6l.predict(X_tr)):.2f} val RMSE={rmse(y_va, m6l.predict(X_va)):.2f}"
      f" | nonzero={int((np.abs(m6l.coef_[1:]) > 1e-12).sum())}/{len(cols)}")
ols = RegresionLineal(X_tr, y_tr).fit_pseudoinversa()
print(f"M5 with city indicator, no penalty: val RMSE={rmse(y_va, ols.predict(X_va)):.2f}")
print("stage 6 val RMSE:", {k: round(v, 2) for k, v in stage6.items()})
print(f"test M1: MAE={mae(y_te, yhat_m1):.2f} RMSE={rmse(y_te, yhat_m1):.2f}")
print(f"test M6-Lasso (post hoc): MAE={mae(y_te, yhat_m6):.2f} RMSE={rmse(y_te, yhat_m6):.2f}")
