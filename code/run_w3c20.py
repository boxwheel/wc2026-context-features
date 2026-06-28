"""
W3C-20: Symmetrized KDE and direct NW multinomial estimation.
  a) Symmetrized KDE: reflect training around elo_diff=0
  b) Direct NW multinomial: P(A,D,H | elo_diff) simultaneously (no two-stage)
  c) Fully non-parametric: NW multinomial + SVM
  d) KDE-TS with KDE Stage 2 (non-parametric H/A within decisive)
  e) Combined: sym-KDE-TS + SVM
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

sys.path.insert(0, os.path.dirname(__file__))
from features import build_match_features, ELO_FEATURES

ARTIFACTS_DIR = "/home/user/research/wave3-context/artifacts"
SEED = 0
N_SPLITS = 5
N_REPEATS = 10
BASELINE_LOSS = 0.8337
FRONTIER_LOSS = 0.7608


class SymKDETwoStageModel:
    """Symmetrized KDE: reflect training data around elo_diff=0 for draw probability."""
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.2):
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        elo = X[:, 0]
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        # Augment with reflected data (elo → -elo)
        elo_aug = np.concatenate([elo, -elo])
        draw_aug = np.concatenate([self.y_draw, self.y_draw])
        self.X_train = elo_aug
        self.y_draw_aug = draw_aug
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
                kde = np.dot(w, self.y_draw_aug) / ws
                p_draw[j] = (1 - self.prior_weight) * np.clip(kde, 0.05, 0.95) + self.prior_weight * self.prior_draw
            else:
                p_draw[j] = self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


class NWMultinomialModel:
    """Direct Nadaraya-Watson for all 3 outcomes simultaneously."""
    def __init__(self, bw=300.0, prior_weight=0.2):
        self.bw = bw
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.n_classes = 3
        # One-hot encode
        self.Y_oh = np.zeros((len(y), 3))
        for i, yi in enumerate(y):
            self.Y_oh[i, yi] = 1.0
        self.prior = self.Y_oh.mean(axis=0)
        return self

    def predict_proba(self, X):
        elo, n = X[:, 0], len(X)
        probs = np.zeros((n, 3))
        for j in range(n):
            w = np.exp(-0.5 * ((elo[j] - self.X_train) / self.bw) ** 2)
            ws = w.sum()
            if ws > 0:
                nw_est = np.dot(w, self.Y_oh) / ws
                probs[j] = (1 - self.prior_weight) * np.clip(nw_est, 0.03, 0.97) + self.prior_weight * self.prior
            else:
                probs[j] = self.prior
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


class KDEFullNPModel:
    """KDE Stage 1 for draw probability + KDE Stage 2 for H/A (both non-parametric)."""
    def __init__(self, bw_s1=300.0, bw_s2=300.0, prior_weight=0.2):
        self.bw_s1 = bw_s1
        self.bw_s2 = bw_s2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        # Stage 2: P(H | decisive) from elo_diff using KDE
        dm = (y != 1)
        self.X_decisive = X[dm, 0].copy()
        self.y_home = (y[dm] == 2).astype(float)
        self.prior_home = self.y_home.mean() if len(self.y_home) > 0 else 0.5
        return self

    def _kde_estimate(self, x_query, x_train, y_val, prior_val, bw):
        n = len(x_query)
        est = np.zeros(n)
        for j in range(n):
            w = np.exp(-0.5 * ((x_query[j] - x_train) / bw) ** 2)
            ws = w.sum()
            if ws > 0:
                est[j] = np.dot(w, y_val) / ws
            else:
                est[j] = prior_val
        return (1 - self.prior_weight) * np.clip(est, 0.05, 0.95) + self.prior_weight * prior_val

    def predict_proba(self, X):
        elo = X[:, 0]
        p_draw = self._kde_estimate(elo, self.X_train, self.y_draw, self.prior_draw, self.bw_s1)
        pdec = 1.0 - p_draw
        if len(self.X_decisive) >= 4:
            p_home_given_dec = self._kde_estimate(elo, self.X_decisive, self.y_home, self.prior_home, self.bw_s2)
        else:
            p_home_given_dec = np.full(len(elo), self.prior_home)
        p = np.stack([pdec * (1 - p_home_given_dec), p_draw, pdec * p_home_given_dec], axis=1)
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

    # ── W3C-20a: Symmetrized KDE-TS + SVM ──────────────────────────────────
    print("\n--- W3C-20a: Symmetrized KDE-TS + SVM (bw sweep) ---")
    best_20a = None
    for bw in [200, 300, 400, 500, 700]:
        m = cv_blend(X, yenc, classes,
            [("SymKDE", lambda bw=bw: SymKDETwoStageModel(bw, 2.0, 0.2)),
             ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-20a_sym_bw{bw}", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_20a is None or m['cv_log_loss_mean'] < best_20a['cv_log_loss_mean']:
            best_20a = m
    print(f"Best W3C-20a: {best_20a['cv_log_loss_mean']:.4f} → {best_20a['experiment']}")

    # ── W3C-20b: Direct NW multinomial + SVM ────────────────────────────────
    print("\n--- W3C-20b: NW Multinomial + SVM (bw sweep) ---")
    best_20b = None
    for bw in [200, 300, 400, 500, 700, 1000]:
        m = cv_blend(X, yenc, classes,
            [("NW", lambda bw=bw: NWMultinomialModel(bw, 0.2)),
             ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-20b_nw_bw{bw}", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_20b is None or m['cv_log_loss_mean'] < best_20b['cv_log_loss_mean']:
            best_20b = m
    print(f"Best W3C-20b: {best_20b['cv_log_loss_mean']:.4f} → {best_20b['experiment']}")

    # ── W3C-20c: NW multinomial alone (no SVM) ──────────────────────────────
    print("\n--- W3C-20c: NW multinomial alone ---")
    best_20c = None
    for bw in [300, 500, 800]:
        m = cv_blend(X, yenc, classes,
            [("NW", lambda bw=bw: NWMultinomialModel(bw, 0.2))],
            [1.0], f"W3C-20c_nw_alone_bw{bw}", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_20c is None or m['cv_log_loss_mean'] < best_20c['cv_log_loss_mean']:
            best_20c = m
    print(f"Best W3C-20c: {best_20c['cv_log_loss_mean']:.4f}")

    # ── W3C-20d: Fully non-parametric KDE (both stages) + SVM ───────────────
    print("\n--- W3C-20d: KDE both stages (bw_s1, bw_s2 sweep) ---")
    best_20d = None
    for bw_s1, bw_s2 in [(300, 300), (300, 500), (300, 800), (400, 400), (500, 300)]:
        m = cv_blend(X, yenc, classes,
            [("KDE_NP", lambda b1=bw_s1, b2=bw_s2: KDEFullNPModel(b1, b2, 0.2)),
             ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-20d_s1{bw_s1}_s2{bw_s2}", verbose=False)
        print(f"  bw_s1={bw_s1}, bw_s2={bw_s2}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_20d is None or m['cv_log_loss_mean'] < best_20d['cv_log_loss_mean']:
            best_20d = m
    print(f"Best W3C-20d: {best_20d['cv_log_loss_mean']:.4f}")

    # ── W3C-20e: Best sym-KDE + NW multinomial + SVM 3-way ──────────────────
    print("\n--- W3C-20e: SymKDE-TS + NW + SVM 3-way ---")
    m20e = cv_blend(X, yenc, classes,
        [("SymKDE", lambda: SymKDETwoStageModel(300.0, 2.0, 0.2)),
         ("NW", lambda: NWMultinomialModel(500.0, 0.2)),
         ("SVM", svm_fn)],
        [1/3, 1/3, 1/3], "W3C-20e_sym_nw_svm")

    print("\nAll W3C-20 experiments done.")


if __name__ == "__main__":
    run_experiments()
