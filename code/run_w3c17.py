"""
W3C-17: New modeling directions beyond KDE hyperparameter tuning.
  a) Combined bw=450 + prior_weight=0.2 (best of each dimension)
  b) 2D KDE for Stage 1 (elo_diff + host_adv)
  c) KDE bandwidth ensemble (average across bw={200,300,450,600})
  d) OOF stacking: LogReg meta-learner on KDE-TS and SVM OOF probs
  e) Epanechnikov kernel instead of Gaussian
  f) Adaptive-bw KDE (bandwidth inversely proportional to local density)
"""

import numpy as np
import pandas as pd
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from scipy.stats import wilcoxon

# ── Constants ──────────────────────────────────────────────────────────────
BASELINE = 0.8337
FRONTIER = 0.7608
ARTDIR   = "/home/user/research/wave3-context/artifacts"
N_SPLITS, N_REPEATS, SEED = 5, 10, 0
os.makedirs(ARTDIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading WC-2026 features...")
data_dir = "/home/user/research/wave3-context/data"
matches  = pd.read_csv(f"{data_dir}/matches.csv")
teams    = pd.read_csv(f"{data_dir}/teams.csv")

elo_map = dict(zip(teams["team_id"], teams["elo_rating"]))
HOST_NATIONS = {"MEX", "USA", "CAN"}

rows = []
for _, r in matches.iterrows():
    h, a = r["home_team_id"], r["away_team_id"]
    eh   = elo_map.get(h, 1500)
    ea   = elo_map.get(a, 1500)
    diff = eh - ea
    host = 1 if h in HOST_NATIONS else (-1 if a in HOST_NATIONS else 0)
    res  = r["result"]
    if res not in ("H", "D", "A"):
        continue
    rows.append({"elo_diff": diff, "host_adv": host, "result": res})

df = pd.DataFrame(rows)
label_map = {"A": 0, "D": 1, "H": 2}
y_raw = df["result"].map(label_map).values
X_raw = df[["elo_diff", "host_adv"]].values
n = len(df)
classes = sorted(df["result"].unique())
print(f"n={n}, classes={classes}")

def get_baseline_losses():
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    bl_losses = []
    for tr, te in rskf.split(X_raw, y_raw):
        Xtr, Xte = X_raw[tr], X_raw[te]
        ytr, yte = y_raw[tr], y_raw[te]
        sc  = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        lr  = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                 multi_class="multinomial", random_state=SEED)
        lr.fit(Xtr_s, ytr)
        bl_losses.append(log_loss(yte, lr.predict_proba(Xte_s)))
    return np.array(bl_losses)

bl_losses = get_baseline_losses()

def verdict(delta, pval):
    if delta < -0.01 and pval < 0.05:
        return "GREEN"
    elif delta > 0.01:
        return "RED"
    return "FLAT"

def save_metrics(exp, mean, std, fold_losses):
    delta = mean - BASELINE
    _, pval = wilcoxon(fold_losses, bl_losses, alternative="less")
    v = verdict(delta, pval)
    os.makedirs(f"{ARTDIR}/{exp}", exist_ok=True)
    m = {
        "experiment": exp,
        "cv_log_loss_mean": round(mean, 4),
        "cv_log_loss_std": round(std, 4),
        "delta_vs_baseline_0.8337": round(delta, 4),
        "delta_vs_frontier_0.7608": round(mean - FRONTIER, 4),
        "wilcoxon_vs_baseline_pvalue": round(pval, 4),
        "verdict_vs_baseline": v,
    }
    with open(f"{ARTDIR}/{exp}/metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    return mean, std, pval, v

def report(exp, mean, std, pval, v):
    print(f"\n{'='*60}")
    print(f"Exp: {exp}")
    print(f"log-loss: {mean:.4f} ± {std:.4f}")
    print(f"Δ base: {mean-BASELINE:+.4f} (p={pval:.4f}) → {v}")


# ── KDE TwoStage model ──────────────────────────────────────────────────────
class KDETwoStageModel:
    def __init__(self, bandwidth=300.0, C2=2.0, prior_weight=0.1, kernel="gaussian"):
        self.bw = bandwidth
        self.C2 = C2
        self.prior_weight = prior_weight
        self.kernel = kernel

    def _kernel_weights(self, diff):
        if self.kernel == "gaussian":
            return np.exp(-0.5 * (diff / self.bw) ** 2)
        elif self.kernel == "epanechnikov":
            u = diff / self.bw
            w = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
            return w
        else:
            return np.exp(-0.5 * (diff / self.bw) ** 2)

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()  # elo_diff (scaled)
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver="lbfgs", max_iter=1000)
            self.stage2.fit(X[dm], (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        elo = X[:, 0]
        n = len(elo)
        p_draw = np.zeros(n)
        for j in range(n):
            w = self._kernel_weights(elo[j] - self.X_train)
            ws = w.sum()
            if ws > 0:
                kde_est = np.dot(w, self.y_draw) / ws
                p_draw[j] = ((1 - self.prior_weight) *
                             np.clip(kde_est, 0.05, 0.95) +
                             self.prior_weight * self.prior_draw)
            else:
                p_draw[j] = self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


class KDE2DTwoStageModel:
    """2D KDE: Stage 1 uses both elo_diff and host_adv."""
    def __init__(self, bw_elo=300.0, bw_host=1.0, C2=2.0, prior_weight=0.2):
        self.bw_elo = bw_elo
        self.bw_host = bw_host
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, :2].copy()  # elo_diff, host_adv (scaled)
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver="lbfgs", max_iter=1000)
            self.stage2.fit(X[dm], (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        n = len(X)
        p_draw = np.zeros(n)
        for j in range(n):
            d_elo  = (X[j, 0] - self.X_train[:, 0]) / self.bw_elo
            d_host = (X[j, 1] - self.X_train[:, 1]) / self.bw_host
            w = np.exp(-0.5 * (d_elo**2 + d_host**2))
            ws = w.sum()
            if ws > 0:
                kde_est = np.dot(w, self.y_draw) / ws
                p_draw[j] = ((1 - self.prior_weight) *
                             np.clip(kde_est, 0.05, 0.95) +
                             self.prior_weight * self.prior_draw)
            else:
                p_draw[j] = self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


class AdaptiveBWKDEModel:
    """KDE with adaptive bandwidth (pilot KDE + local density-based scaling)."""
    def __init__(self, pilot_bw=300.0, C2=2.0, prior_weight=0.2, alpha=0.5):
        self.pilot_bw = pilot_bw
        self.C2 = C2
        self.prior_weight = prior_weight
        self.alpha = alpha  # sensitivity to local density

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        # Compute pilot densities for adaptive bw
        n = len(self.X_train)
        pilot_dens = np.zeros(n)
        for i in range(n):
            w = np.exp(-0.5 * ((self.X_train[i] - self.X_train) / self.pilot_bw)**2)
            pilot_dens[i] = w.sum() / (n * self.pilot_bw)
        g = np.exp(np.log(pilot_dens).mean())  # geometric mean
        self.lambdas = (pilot_dens / g) ** (-self.alpha)  # local bw multipliers
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver="lbfgs", max_iter=1000)
            self.stage2.fit(X[dm], (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        elo = X[:, 0]
        n = len(elo)
        p_draw = np.zeros(n)
        for j in range(n):
            local_bw = self.pilot_bw * self.lambdas
            w = np.exp(-0.5 * ((elo[j] - self.X_train) / local_bw)**2)
            ws = w.sum()
            if ws > 0:
                kde_est = np.dot(w, self.y_draw) / ws
                p_draw[j] = ((1 - self.prior_weight) *
                             np.clip(kde_est, 0.05, 0.95) +
                             self.prior_weight * self.prior_draw)
            else:
                p_draw[j] = self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


def cv_blend(models_fn, weights, exp_name):
    """CV with a list of model factory functions and blend weights."""
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    fold_losses = []
    for tr, te in rskf.split(X_raw, y_raw):
        Xtr, Xte = X_raw[tr], X_raw[te]
        ytr, yte = y_raw[tr], y_raw[te]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        blend = np.zeros((len(te), 3))
        for fn, w in zip(models_fn, weights):
            m = fn()
            m.fit(Xtr_s, ytr)
            blend += w * m.predict_proba(Xte_s)
        fold_losses.append(log_loss(yte, blend))
    arr = np.array(fold_losses)
    mean, std = arr.mean(), arr.std()
    mean, std, pval, v = save_metrics(exp_name, mean, std, arr)
    report(exp_name, mean, std, pval, v)
    return mean, v


# ── W3C-17a: Combined bw=450 + prior_weight=0.2 ────────────────────────────
print("\n--- W3C-17a: KDE-TS(bw=450, pw=0.2) + SVM ---")
svm_fn  = lambda: SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED)
kde17a  = lambda: KDETwoStageModel(bandwidth=450.0, C2=2.0, prior_weight=0.2)
mean17a, v17a = cv_blend([kde17a, svm_fn], [0.5, 0.5], "W3C-17a_bw450_pw02")

# ── W3C-17b: 2D KDE for Stage 1 (elo_diff + host_adv sweep) ───────────────
print("\n--- W3C-17b: 2D KDE Stage 1 (bw_host sweep) ---")
best_17b, best_bw_host = 999, None
for bw_host in [0.3, 0.5, 1.0, 2.0, 5.0]:
    kde2d = lambda bh=bw_host: KDE2DTwoStageModel(bw_elo=450.0, bw_host=bh, C2=2.0, prior_weight=0.2)
    svm_f = lambda: SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED)
    m, v = cv_blend([kde2d, svm_f], [0.5, 0.5], f"W3C-17b_bwh{str(bw_host).replace('.','')}")
    arrow = "→ " + v
    print(f"  bw_host={bw_host}: {m:.4f} {arrow}")
    if m < best_17b:
        best_17b, best_bw_host = m, bw_host
print(f"Best W3C-17b: {best_17b:.4f} → bw_host={best_bw_host}")

# ── W3C-17c: KDE bandwidth ensemble (average across multiple bw) ───────────
print("\n--- W3C-17c: KDE bandwidth ensemble (bw={200,300,450,600}) ---")
bw_ensemble = [200, 300, 450, 600]
models_fn_ens = [lambda bw=bw: KDETwoStageModel(bandwidth=float(bw), C2=2.0, prior_weight=0.2)
                 for bw in bw_ensemble]
# Blend: equal weight across ensemble members + SVM
kde_w = 0.5 / len(bw_ensemble)
ens_fns  = models_fn_ens + [svm_fn]
ens_wts  = [kde_w] * len(bw_ensemble) + [0.5]
mean17c, v17c = cv_blend(ens_fns, ens_wts, "W3C-17c_bw_ensemble")
print(f"Ensemble result: {mean17c:.4f} → {v17c}")

# ── W3C-17d: Epanechnikov kernel ────────────────────────────────────────────
print("\n--- W3C-17d: Epanechnikov kernel KDE-TS + SVM ---")
best_17d = 999
for bw in [400, 500, 600, 700, 800]:
    kde_ep = lambda b=bw: KDETwoStageModel(bandwidth=float(b), C2=2.0,
                                            prior_weight=0.2, kernel="epanechnikov")
    m, v = cv_blend([kde_ep, svm_fn], [0.5, 0.5], f"W3C-17d_ep_bw{b}")
    print(f"  bw={b}: {m:.4f} → {v}")
    if m < best_17d:
        best_17d = m
print(f"Best W3C-17d: {best_17d:.4f}")

# ── W3C-17e: Adaptive bandwidth KDE ─────────────────────────────────────────
print("\n--- W3C-17e: Adaptive-bandwidth KDE-TS + SVM ---")
best_17e = 999
for alpha in [0.3, 0.5, 0.7]:
    for pilot_bw in [300, 450]:
        kde_ad = lambda a=alpha, pb=pilot_bw: AdaptiveBWKDEModel(
            pilot_bw=float(pb), C2=2.0, prior_weight=0.2, alpha=a)
        tag = f"W3C-17e_a{str(alpha).replace('.','')}_pb{pilot_bw}"
        m, v = cv_blend([kde_ad, svm_fn], [0.5, 0.5], tag)
        print(f"  alpha={alpha}, pilot_bw={pilot_bw}: {m:.4f} → {v}")
        if m < best_17e:
            best_17e = m
print(f"Best W3C-17e: {best_17e:.4f}")

# ── W3C-17f: OOF stacking (KDE-TS + SVM → LogReg meta) ────────────────────
print("\n--- W3C-17f: OOF stacking with LogReg meta-learner ---")

def cv_oof_stack():
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    fold_losses = []
    for tr, te in rskf.split(X_raw, y_raw):
        Xtr, Xte = X_raw[tr], X_raw[te]
        ytr, yte = y_raw[tr], y_raw[te]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

        # Inner CV for OOF predictions on training fold
        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED+1)
        oof_preds_tr = np.zeros((len(tr), 6))  # 3 from KDE-TS, 3 from SVM

        for itr, ival in inner_cv.split(Xtr_s, ytr):
            Xi_tr, Xi_val = Xtr_s[itr], Xtr_s[ival]
            yi_tr = ytr[itr]
            kde = KDETwoStageModel(bandwidth=450.0, C2=2.0, prior_weight=0.2)
            svm = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED)
            kde.fit(Xi_tr, yi_tr)
            svm.fit(Xi_tr, yi_tr)
            oof_preds_tr[ival, :3] += kde.predict_proba(Xi_val)
            oof_preds_tr[ival, 3:] += svm.predict_proba(Xi_val)

        # Average across repeats (each sample appears n_repeats*2=2 times)
        # Normalize so they sum to 1
        oof_preds_tr[:, :3] /= oof_preds_tr[:, :3].sum(axis=1, keepdims=True)
        oof_preds_tr[:, 3:] /= oof_preds_tr[:, 3:].sum(axis=1, keepdims=True)

        # Train meta-learner on OOF predictions (no scaling needed)
        meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                   multi_class="multinomial", random_state=SEED)
        meta.fit(oof_preds_tr, ytr)

        # For test: fit base models on full training fold, predict, then meta
        kde_full = KDETwoStageModel(bandwidth=450.0, C2=2.0, prior_weight=0.2)
        svm_full = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED)
        kde_full.fit(Xtr_s, ytr)
        svm_full.fit(Xtr_s, ytr)
        test_feats = np.hstack([kde_full.predict_proba(Xte_s),
                                svm_full.predict_proba(Xte_s)])
        final_preds = meta.predict_proba(test_feats)
        fold_losses.append(log_loss(yte, final_preds))

    arr = np.array(fold_losses)
    mean, std = arr.mean(), arr.std()
    return mean, std, arr

mean17f, std17f, arr17f = cv_oof_stack()
mean17f, std17f, pval17f, v17f = save_metrics("W3C-17f_oof_stack", mean17f, std17f, arr17f)
report("W3C-17f_oof_stack", mean17f, std17f, pval17f, v17f)

print("\nAll W3C-17 experiments done.")
