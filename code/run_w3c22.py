"""
W3C-22: Calibration + alternative kernels + LOOCV bandwidth.
  a) Temperature scaling on blend output (post-hoc calibration)
  b) Laplace/triangular kernel for KDE Stage 1
  c) LOOCV bandwidth selection per fold (data-adaptive bw)
  d) KDE-TS + LogReg(elo) + SVM three-way (adding LogReg for diversity)
  e) Shrinkage blend: mix model predictions with multinomial prior
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import wilcoxon
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(__file__))
from features import build_match_features, ELO_FEATURES

ARTIFACTS_DIR = "/home/user/research/wave3-context/artifacts"
SEED = 0
N_SPLITS = 5
N_REPEATS = 10
BASELINE_LOSS = 0.8337
FRONTIER_LOSS = 0.7608


class KDETwoStageModel:
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.2):
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=1000)
            self.stage2.fit(X[dm], (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        elo, n = X[:, 0], len(X)
        p_draw = np.zeros(n)
        for j in range(n):
            w = np.exp(-0.5 * ((elo[j] - self.X_train) / self.bw) ** 2)
            ws = w.sum()
            if ws > 0:
                kde = np.dot(w, self.y_draw) / ws
                p_draw[j] = (1 - self.prior_weight) * np.clip(kde, 0.05, 0.95) + self.prior_weight * self.prior_draw
            else:
                p_draw[j] = self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


class LaplaceKDETwoStageModel:
    """KDE Stage 1 using Laplace (double-exponential) kernel."""
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.2):
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=1000)
            self.stage2.fit(X[dm], (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        elo, n = X[:, 0], len(X)
        p_draw = np.zeros(n)
        for j in range(n):
            # Laplace kernel: exp(-|elo_diff| / bw)
            w = np.exp(-np.abs(elo[j] - self.X_train) / self.bw)
            ws = w.sum()
            if ws > 0:
                kde = np.dot(w, self.y_draw) / ws
                p_draw[j] = (1 - self.prior_weight) * np.clip(kde, 0.05, 0.95) + self.prior_weight * self.prior_draw
            else:
                p_draw[j] = self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


class LOOCVKDETwoStageModel:
    """KDE Stage 1 with bandwidth chosen by LOOCV on training set."""
    def __init__(self, bw_candidates=None, C2=2.0, prior_weight=0.2):
        self.bw_candidates = bw_candidates or [100, 200, 300, 400, 500, 700]
        self.C2 = C2
        self.prior_weight = prior_weight

    def _loocv_loss(self, bw, elo, y_draw):
        n = len(elo)
        total = 0.0
        for i in range(n):
            x_loo = np.delete(elo, i)
            y_loo = np.delete(y_draw, i)
            w = np.exp(-0.5 * ((elo[i] - x_loo) / bw) ** 2)
            ws = w.sum()
            if ws > 0:
                p_d = np.dot(w, y_loo) / ws
            else:
                p_d = y_draw.mean()
            p_d = np.clip(p_d, 0.01, 0.99)
            # LOO log-loss for this sample
            true = y_draw[i]
            total += -(true * np.log(p_d) + (1 - true) * np.log(1 - p_d))
        return total / n

    def fit(self, X, y):
        elo = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        # Pick best bw by LOOCV
        best_bw, best_loss = self.bw_candidates[0], np.inf
        for bw in self.bw_candidates:
            loss = self._loocv_loss(bw, elo, self.y_draw)
            if loss < best_loss:
                best_loss = loss
                best_bw = bw
        self.best_bw = best_bw
        self.X_train = elo
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=1000)
            self.stage2.fit(X[dm], (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        elo, n = X[:, 0], len(X)
        p_draw = np.zeros(n)
        for j in range(n):
            w = np.exp(-0.5 * ((elo[j] - self.X_train) / self.best_bw) ** 2)
            ws = w.sum()
            if ws > 0:
                kde = np.dot(w, self.y_draw) / ws
                p_draw[j] = (1 - self.prior_weight) * np.clip(kde, 0.05, 0.95) + self.prior_weight * self.prior_draw
            else:
                p_draw[j] = self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


def cv_blend(X, y_enc, classes, components, blend_weights, exp_name, verbose=True):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    n, nc = len(y_enc), len(classes)
    oof = np.zeros((n, nc))
    fl = []
    for ti, vi in cv.split(X, y_enc):
        Xtr, Xv = X[ti], X[vi]
        ytr, yv = y_enc[ti], y_enc[vi]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xv  = sc.transform(Xv)
        bp = np.zeros((len(vi), nc))
        for (nm, mf), w in zip(components, blend_weights):
            m = mf(); m.fit(Xtr, ytr); bp += w * m.predict_proba(Xv)
        oof[vi] += bp / N_REPEATS
        fl.append(log_loss(yv, bp))
    ra = np.array(fl).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    ml, sl = ra.mean(), ra.std()
    pm = [-np.log(oof[i, y_enc[i]] + 1e-15) for i in range(n)]
    try: _, pb = wilcoxon(np.array(pm) - BASELINE_LOSS, alternative='less')
    except: pb = 1.0
    try: _, pf = wilcoxon(np.array(pm) - FRONTIER_LOSS, alternative='less')
    except: pf = 1.0
    db = ml - BASELINE_LOSS
    v = "GREEN" if db < -0.01 and pb < 0.05 else ("RED" if db > 0.01 else "FLAT")
    acc = accuracy_score(y_enc, np.argmax(oof, axis=1))
    met = {"experiment": exp_name, "cv_log_loss_mean": round(ml, 4), "cv_log_loss_std": round(sl, 4),
           "accuracy": round(acc, 4), "delta_vs_baseline_0.8337": round(db, 4),
           "delta_vs_frontier_0.7608": round(ml - FRONTIER_LOSS, 4),
           "wilcoxon_vs_baseline_pvalue": round(pb, 4), "wilcoxon_vs_frontier_pvalue": round(pf, 4),
           "verdict_vs_baseline": v, "label_classes": list(classes),
           "blend_weights": blend_weights, "blend_components": [nm for nm, _ in components]}
    if verbose:
        print(f"\n{'='*60}\nExp: {exp_name}\nlog-loss: {ml:.4f} ± {sl:.4f}\nΔ base: {db:+.4f} (p={pb:.4f}) → {v}")
    od = os.path.join(ARTIFACTS_DIR, exp_name)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "metrics.json"), "w") as f:
        json.dump(met, f, indent=2)
    return met


def cv_blend_with_temp(X, y_enc, classes, components, blend_weights, exp_name, verbose=True):
    """Same as cv_blend but applies temperature scaling to the blend output."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    n, nc = len(y_enc), len(classes)
    oof = np.zeros((n, nc))
    fl = []
    for ti, vi in cv.split(X, y_enc):
        Xtr, Xv = X[ti], X[vi]
        ytr, yv = y_enc[ti], y_enc[vi]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xv  = sc.transform(Xv)
        # Get raw blend predictions on val
        bp_raw = np.zeros((len(vi), nc))
        for (nm, mf), w in zip(components, blend_weights):
            m = mf(); m.fit(Xtr, ytr); bp_raw += w * m.predict_proba(Xv)
        # Temperature scaling: find T that minimizes log-loss on train fold OOF
        # Use train-fold val to find T (inner split)
        # Simple: optimize T on training fold proba
        # For simplicity, use OOF from training data for T estimation
        bp_tr = np.zeros((len(ti), nc))
        inner_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=SEED)
        inner_oof = np.zeros((len(ti), nc))
        for tii, vii in inner_cv.split(Xtr, ytr):
            inn_bp = np.zeros((len(vii), nc))
            for (nm, mf), w in zip(components, blend_weights):
                m2 = mf(); m2.fit(Xtr[tii], ytr[tii]); inn_bp += w * m2.predict_proba(Xtr[vii])
            inner_oof[vii] += inn_bp / 2
        # Find temperature
        def temp_loss(T):
            scaled = inner_oof ** (1/T)
            scaled /= scaled.sum(axis=1, keepdims=True)
            return log_loss(ytr, scaled)
        res = minimize_scalar(temp_loss, bounds=(0.5, 3.0), method='bounded')
        T = res.x
        # Apply temperature
        bp = bp_raw ** (1 / T)
        bp /= bp.sum(axis=1, keepdims=True)
        oof[vi] += bp / N_REPEATS
        fl.append(log_loss(yv, bp))
    ra = np.array(fl).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    ml, sl = ra.mean(), ra.std()
    pm = [-np.log(oof[i, y_enc[i]] + 1e-15) for i in range(n)]
    try: _, pb = wilcoxon(np.array(pm) - BASELINE_LOSS, alternative='less')
    except: pb = 1.0
    try: _, pf = wilcoxon(np.array(pm) - FRONTIER_LOSS, alternative='less')
    except: pf = 1.0
    db = ml - BASELINE_LOSS
    v = "GREEN" if db < -0.01 and pb < 0.05 else ("RED" if db > 0.01 else "FLAT")
    acc = accuracy_score(y_enc, np.argmax(oof, axis=1))
    met = {"experiment": exp_name, "cv_log_loss_mean": round(ml, 4), "cv_log_loss_std": round(sl, 4),
           "accuracy": round(acc, 4), "delta_vs_baseline_0.8337": round(db, 4),
           "delta_vs_frontier_0.7608": round(ml - FRONTIER_LOSS, 4),
           "wilcoxon_vs_baseline_pvalue": round(pb, 4), "wilcoxon_vs_frontier_pvalue": round(pf, 4),
           "verdict_vs_baseline": v, "label_classes": list(classes),
           "blend_weights": blend_weights, "blend_components": [nm for nm, _ in components],
           "method": "temperature_scaling"}
    if verbose:
        print(f"\n{'='*60}\nExp: {exp_name}\nlog-loss: {ml:.4f} ± {sl:.4f}\nΔ base: {db:+.4f} (p={pb:.4f}) → {v}")
    od = os.path.join(ARTIFACTS_DIR, exp_name)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "metrics.json"), "w") as f:
        json.dump(met, f, indent=2)
    return met


def run_experiments():
    print("Loading WC-2026 features...")
    df = build_match_features(include_context=False)
    X = df[ELO_FEATURES].values
    y = df["label"].values
    le = LabelEncoder()
    yenc = le.fit_transform(y)
    classes = le.classes_
    print(f"n={len(yenc)}, classes={classes}")

    kde_best = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)
    svm_fn   = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
    lr_fn    = lambda: LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)

    # ── W3C-22a: Temperature scaling on best KDE-TS + SVM blend ─────────────
    print("\n--- W3C-22a: Temperature scaling on KDE-TS + SVM ---")
    m22a = cv_blend_with_temp(X, yenc, classes,
        [("KDE_TS", kde_best), ("SVM", svm_fn)],
        [0.5, 0.5], "W3C-22a_temp_scaling")

    # ── W3C-22b: Laplace kernel KDE-TS + SVM (bw sweep) ────────────────────
    print("\n--- W3C-22b: Laplace kernel KDE + SVM ---")
    best_22b = None
    for bw in [100, 200, 300, 500, 800]:
        laplace_fn = lambda bw=bw: LaplaceKDETwoStageModel(bw, 2.0, 0.2)
        m = cv_blend(X, yenc, classes,
            [("LaplaceKDE", laplace_fn), ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-22b_laplace_bw{bw}", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_22b is None or m['cv_log_loss_mean'] < best_22b['cv_log_loss_mean']:
            best_22b = m
    print(f"Best W3C-22b: {best_22b['cv_log_loss_mean']:.4f}")

    # ── W3C-22c: LOOCV bandwidth selection per fold ──────────────────────────
    print("\n--- W3C-22c: LOOCV bandwidth + SVM ---")
    loocv_fn = lambda: LOOCVKDETwoStageModel(bw_candidates=[100, 200, 300, 400, 500, 700], C2=2.0, prior_weight=0.2)
    m22c = cv_blend(X, yenc, classes,
        [("LOOCV_KDE", loocv_fn), ("SVM", svm_fn)],
        [0.5, 0.5], "W3C-22c_loocv_bw")

    # ── W3C-22d: KDE-TS + SVM + LogReg(elo) three-way ───────────────────────
    print("\n--- W3C-22d: KDE-TS + SVM + LogReg(elo) three-way ---")
    best_22d = None
    for w_kde, w_svm, w_lr in [(1/3, 1/3, 1/3), (0.4, 0.4, 0.2), (0.5, 0.3, 0.2), (0.4, 0.3, 0.3)]:
        tag = f"k{int(w_kde*10)}_s{int(w_svm*10)}_l{int(w_lr*10)}"
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best), ("SVM", svm_fn), ("LR", lr_fn)],
            [w_kde, w_svm, w_lr], f"W3C-22d_3way_{tag}", verbose=False)
        print(f"  (KDE={w_kde:.1f}, SVM={w_svm:.1f}, LR={w_lr:.1f}): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_22d is None or m['cv_log_loss_mean'] < best_22d['cv_log_loss_mean']:
            best_22d = m
    print(f"Best W3C-22d: {best_22d['cv_log_loss_mean']:.4f}")

    # ── W3C-22e: Prior shrinkage blend (mix with uniform prior) ─────────────
    print("\n--- W3C-22e: Prior shrinkage (shrink toward uniform prior) ---")
    # Pure KDE-TS + SVM blend, then shrink toward P=(0.33,0.33,0.33)
    # Implemented by adding a tiny weight for a "uniform predictor"
    class UniformPredictor:
        def fit(self, X, y): return self
        def predict_proba(self, X): return np.full((len(X), 3), 1/3)
    uniform_fn = lambda: UniformPredictor()
    best_22e = None
    for eps in [0.02, 0.05, 0.1, 0.15]:
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best), ("SVM", svm_fn), ("Uniform", uniform_fn)],
            [0.5*(1-eps), 0.5*(1-eps), eps], f"W3C-22e_shrink{int(eps*100):02d}", verbose=False)
        print(f"  eps={eps}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_22e is None or m['cv_log_loss_mean'] < best_22e['cv_log_loss_mean']:
            best_22e = m
    print(f"Best W3C-22e: {best_22e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-22 experiments done.")
    all_bests = [m22a, best_22b, m22c, best_22d, best_22e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-22 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")


if __name__ == "__main__":
    run_experiments()
