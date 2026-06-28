"""
W3C-24: Kernel mixture, Cauchy/Student-t KDE, adaptive Stage 2.
  a) Gaussian + Laplace KDE blend for Stage 1 (two-kernel mixture)
  b) Cauchy kernel (polynomial decay) for Stage 1
  c) Student-t kernel (ν=2, 3, 5) for Stage 1
  d) LogisticRegressionCV for Stage 2 (adaptive C)
  e) SVM binary classifier for Stage 2 (H vs A within decisive)
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from features import build_match_features, ELO_FEATURES

ARTIFACTS_DIR = "/home/user/research/wave3-context/artifacts"
SEED = 0
N_SPLITS = 5
N_REPEATS = 10
BASELINE_LOSS = 0.8337
FRONTIER_LOSS = 0.7608


class GaussianLaplaceBlendKDEModel:
    """Stage 1: weighted blend of Gaussian + Laplace kernel estimates."""
    def __init__(self, bw_g=300.0, bw_l=200.0, alpha=0.5, C2=2.0, prior_weight=0.2):
        self.bw_g = bw_g
        self.bw_l = bw_l
        self.alpha = alpha  # weight on Gaussian
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

    def _kde(self, elo, kernel='gaussian'):
        n = len(elo)
        p_draw = np.zeros(n)
        for j in range(n):
            if kernel == 'gaussian':
                w = np.exp(-0.5 * ((elo[j] - self.X_train) / self.bw_g) ** 2)
            else:  # laplace
                w = np.exp(-np.abs(elo[j] - self.X_train) / self.bw_l)
            ws = w.sum()
            if ws > 0:
                p_draw[j] = np.dot(w, self.y_draw) / ws
            else:
                p_draw[j] = self.prior_draw
        return p_draw

    def predict_proba(self, X):
        elo, n = X[:, 0], len(X)
        p_g = self._kde(elo, 'gaussian')
        p_l = self._kde(elo, 'laplace')
        p_raw = self.alpha * p_g + (1 - self.alpha) * p_l
        p_draw = (1 - self.prior_weight) * np.clip(p_raw, 0.05, 0.95) + self.prior_weight * self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


class CauchyKDETwoStageModel:
    """KDE Stage 1 with Cauchy kernel: K(x) = 1/(1 + (x/bw)^2)."""
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
            w = 1.0 / (1.0 + ((elo[j] - self.X_train) / self.bw) ** 2)
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


class StudentTKDETwoStageModel:
    """KDE Stage 1 with Student-t kernel: K(x) ∝ (1 + (x/bw)^2/nu)^(-(nu+1)/2)."""
    def __init__(self, bw=300.0, nu=3, C2=2.0, prior_weight=0.2):
        self.bw = bw
        self.nu = nu
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
            t2 = ((elo[j] - self.X_train) / self.bw) ** 2
            w = (1.0 + t2 / self.nu) ** (-(self.nu + 1) / 2.0)
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


class KDETwStageWithCVStage2:
    """KDE Stage 1 + LogisticRegressionCV for Stage 2 (data-adaptive C)."""
    def __init__(self, bw=300.0, prior_weight=0.2):
        self.bw = bw
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegressionCV(
                Cs=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0], cv=3, solver='lbfgs', max_iter=1000)
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


class KDETwoStageSVMStage2:
    """KDE Stage 1 + binary SVM for Stage 2 (H vs A within decisive)."""
    def __init__(self, bw=300.0, C2=1.0, prior_weight=0.2):
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = SVC(C=self.C2, kernel='rbf', gamma='scale', probability=True)
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


def run_experiments():
    print("Loading WC-2026 features...")
    df = build_match_features(include_context=False)
    X = df[ELO_FEATURES].values
    y = df["label"].values
    le = LabelEncoder()
    yenc = le.fit_transform(y)
    classes = le.classes_
    print(f"n={len(yenc)}, classes={classes}")

    svm_fn = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)

    # ── W3C-24a: Gaussian+Laplace blend Stage 1 + SVM ───────────────────────
    print("\n--- W3C-24a: Gaussian+Laplace Stage 1 blend + SVM ---")
    best_24a = None
    for alpha_g in [0.3, 0.5, 0.7]:
        for bw_l in [150, 200, 300]:
            gl_fn = lambda ag=alpha_g, bl=bw_l: GaussianLaplaceBlendKDEModel(bw_g=300, bw_l=bl, alpha=ag)
            m = cv_blend(X, yenc, classes,
                [("GL_KDE", gl_fn), ("SVM", svm_fn)],
                [0.5, 0.5], f"W3C-24a_ag{int(alpha_g*10)}_bl{bw_l}", verbose=False)
            print(f"  alpha_g={alpha_g}, bw_l={bw_l}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
            if best_24a is None or m['cv_log_loss_mean'] < best_24a['cv_log_loss_mean']:
                best_24a = m
    print(f"Best W3C-24a: {best_24a['cv_log_loss_mean']:.4f}")

    # ── W3C-24b: Cauchy kernel KDE + SVM ────────────────────────────────────
    print("\n--- W3C-24b: Cauchy kernel KDE + SVM ---")
    best_24b = None
    for bw in [100, 200, 300, 500, 800]:
        cauchy_fn = lambda bw=bw: CauchyKDETwoStageModel(bw, 2.0, 0.2)
        m = cv_blend(X, yenc, classes,
            [("CauchyKDE", cauchy_fn), ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-24b_cauchy_bw{bw}", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_24b is None or m['cv_log_loss_mean'] < best_24b['cv_log_loss_mean']:
            best_24b = m
    print(f"Best W3C-24b: {best_24b['cv_log_loss_mean']:.4f}")

    # ── W3C-24c: Student-t kernel KDE + SVM ─────────────────────────────────
    print("\n--- W3C-24c: Student-t kernel KDE + SVM ---")
    best_24c = None
    for nu in [2, 3, 5, 10]:
        for bw in [200, 300, 500]:
            st_fn = lambda nu=nu, bw=bw: StudentTKDETwoStageModel(bw, nu, 2.0, 0.2)
            m = cv_blend(X, yenc, classes,
                [("StKDE", st_fn), ("SVM", svm_fn)],
                [0.5, 0.5], f"W3C-24c_nu{nu}_bw{bw}", verbose=False)
            print(f"  nu={nu}, bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
            if best_24c is None or m['cv_log_loss_mean'] < best_24c['cv_log_loss_mean']:
                best_24c = m
    print(f"Best W3C-24c: {best_24c['cv_log_loss_mean']:.4f}")

    # ── W3C-24d: KDE-TS with LogRegressionCV Stage 2 + SVM ──────────────────
    print("\n--- W3C-24d: KDE-TS + LogRegCV Stage 2 + SVM ---")
    kdelrcv_fn = lambda: KDETwStageWithCVStage2(bw=300.0, prior_weight=0.2)
    m24d = cv_blend(X, yenc, classes,
        [("KDE_LRCV", kdelrcv_fn), ("SVM", svm_fn)],
        [0.5, 0.5], "W3C-24d_kde_lrcv")

    # ── W3C-24e: KDE-TS with SVM Stage 2 + outer SVM ────────────────────────
    print("\n--- W3C-24e: KDE-TS with SVM Stage 2 (binary) + outer SVM ---")
    best_24e = None
    for C2 in [0.5, 1.0, 2.0]:
        kde_svm_s2 = lambda C2=C2: KDETwoStageSVMStage2(bw=300.0, C2=C2, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("KDE_SVMS2", kde_svm_s2), ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-24e_svms2_C{int(C2*10)}", verbose=False)
        print(f"  C2={C2}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_24e is None or m['cv_log_loss_mean'] < best_24e['cv_log_loss_mean']:
            best_24e = m
    print(f"Best W3C-24e: {best_24e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-24 experiments done.")
    all_bests = [best_24a, best_24b, best_24c, m24d, best_24e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-24 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")


if __name__ == "__main__":
    run_experiments()
