"""
W3C-23: Bootstrap ensemble of KDE-TS + advanced blend strategies.
  a) Bagged KDE-TS: average over bootstrap samples of training data
  b) Bayesian bandwidth averaging: weight KDE-TS predictions by bw likelihood
  c) Blend weight optimization via Nelder-Mead (3-component weight search)
  d) Poly-SVM + KDE-TS blend
  e) Multi-model mega-ensemble: KDE-TS + SVM + RF(best) + LR
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import wilcoxon
from scipy.optimize import minimize

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


class BaggedKDETwoStageModel:
    """Bootstrap ensemble of KDE-TS: reduce variance via bagging."""
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.2, n_bags=50, seed=0):
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight
        self.n_bags = n_bags
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n = len(y)
        self.bags = []
        for _ in range(self.n_bags):
            idx = rng.choice(n, n, replace=True)
            m = KDETwoStageModel(self.bw, self.C2, self.prior_weight)
            m.fit(X[idx], y[idx])
            self.bags.append(m)
        return self

    def predict_proba(self, X):
        probs = np.array([m.predict_proba(X) for m in self.bags])
        return probs.mean(axis=0)


class BayesianBWKDETwoStageModel:
    """Bayesian bandwidth average: weight bw predictions by LOO likelihood."""
    def __init__(self, bw_candidates=None, C2=2.0, prior_weight=0.2):
        self.bw_candidates = bw_candidates or [100, 200, 300, 400, 500, 700, 1000]
        self.C2 = C2
        self.prior_weight = prior_weight

    def _loo_log_lik(self, bw, elo, y_draw):
        n = len(elo)
        ll = 0.0
        prior = y_draw.mean()
        for i in range(n):
            x_loo = np.delete(elo, i)
            y_loo = np.delete(y_draw, i)
            w = np.exp(-0.5 * ((elo[i] - x_loo) / bw) ** 2)
            ws = w.sum()
            p_d = np.dot(w, y_loo) / ws if ws > 0 else prior
            p_d = (1 - self.prior_weight) * np.clip(p_d, 0.05, 0.95) + self.prior_weight * prior
            yi = y_draw[i]
            ll += yi * np.log(p_d + 1e-15) + (1 - yi) * np.log(1 - p_d + 1e-15)
        return ll

    def fit(self, X, y):
        elo = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        # Compute LOO log-likelihood for each bw
        lls = np.array([self._loo_log_lik(bw, elo, self.y_draw) for bw in self.bw_candidates])
        # Softmax weights
        lls_shifted = lls - lls.max()
        self.bw_weights = np.exp(lls_shifted) / np.exp(lls_shifted).sum()
        self.X_train = elo
        # Train stage 2 models for each bw candidate
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
        for bw, w_bw in zip(self.bw_candidates, self.bw_weights):
            p_d_bw = np.zeros(n)
            for j in range(n):
                w = np.exp(-0.5 * ((elo[j] - self.X_train) / bw) ** 2)
                ws = w.sum()
                if ws > 0:
                    kde = np.dot(w, self.y_draw) / ws
                    p_d_bw[j] = (1 - self.prior_weight) * np.clip(kde, 0.05, 0.95) + self.prior_weight * self.prior_draw
                else:
                    p_d_bw[j] = self.prior_draw
            p_draw += w_bw * p_d_bw
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

    kde_best = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)
    svm_fn   = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
    lr_fn    = lambda: LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)

    # ── W3C-23a: Bagged KDE-TS + SVM ────────────────────────────────────────
    print("\n--- W3C-23a: Bagged KDE-TS + SVM (n_bags sweep) ---")
    best_23a = None
    for nb in [20, 50, 100]:
        bagged_fn = lambda nb=nb: BaggedKDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2, n_bags=nb, seed=SEED)
        m = cv_blend(X, yenc, classes,
            [("BaggedKDE", bagged_fn), ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-23a_bag{nb}", verbose=False)
        print(f"  n_bags={nb}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_23a is None or m['cv_log_loss_mean'] < best_23a['cv_log_loss_mean']:
            best_23a = m
    print(f"Best W3C-23a: {best_23a['cv_log_loss_mean']:.4f}")

    # ── W3C-23b: Bayesian bandwidth averaging + SVM ──────────────────────────
    print("\n--- W3C-23b: Bayesian BW averaging + SVM ---")
    bayes_fn = lambda: BayesianBWKDETwoStageModel(
        bw_candidates=[100, 200, 300, 400, 500, 700, 1000], C2=2.0, prior_weight=0.2)
    m23b = cv_blend(X, yenc, classes,
        [("BayesKDE", bayes_fn), ("SVM", svm_fn)],
        [0.5, 0.5], "W3C-23b_bayesian_bw")

    # ── W3C-23c: Bagged KDE-TS alone (no SVM) ───────────────────────────────
    print("\n--- W3C-23c: Bagged KDE-TS alone ---")
    bagged100_fn = lambda: BaggedKDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2, n_bags=100, seed=SEED)
    m23c = cv_blend(X, yenc, classes,
        [("BaggedKDE", bagged100_fn)],
        [1.0], "W3C-23c_bagged_alone")

    # ── W3C-23d: Polynomial SVM + KDE-TS ─────────────────────────────────────
    print("\n--- W3C-23d: Poly SVM + KDE-TS ---")
    best_23d = None
    for deg in [2, 3]:
        poly_fn = lambda deg=deg: SVC(C=1.0, kernel='poly', degree=deg, probability=True)
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best), ("PolySVM", poly_fn)],
            [0.5, 0.5], f"W3C-23d_poly_deg{deg}", verbose=False)
        print(f"  poly_degree={deg}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_23d is None or m['cv_log_loss_mean'] < best_23d['cv_log_loss_mean']:
            best_23d = m
    print(f"Best W3C-23d: {best_23d['cv_log_loss_mean']:.4f}")

    # ── W3C-23e: KDE-TS + SVM + LR + RF mega-ensemble ───────────────────────
    print("\n--- W3C-23e: Mega ensemble KDE+SVM+LR+RF ---")
    rf_fn = lambda: RandomForestClassifier(n_estimators=200, max_depth=5, random_state=SEED, n_jobs=-1)
    best_23e = None
    for w_kde, w_svm, w_lr, w_rf in [
        (0.4, 0.3, 0.15, 0.15),
        (0.35, 0.35, 0.15, 0.15),
        (0.4, 0.25, 0.2, 0.15),
        (0.3, 0.3, 0.2, 0.2),
    ]:
        tag = f"k{int(w_kde*100)}_s{int(w_svm*100)}"
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best), ("SVM", svm_fn), ("LR", lr_fn), ("RF", rf_fn)],
            [w_kde, w_svm, w_lr, w_rf], f"W3C-23e_mega_{tag}", verbose=False)
        print(f"  (KDE={w_kde}, SVM={w_svm}, LR={w_lr}, RF={w_rf}): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_23e is None or m['cv_log_loss_mean'] < best_23e['cv_log_loss_mean']:
            best_23e = m
    print(f"Best W3C-23e: {best_23e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-23 experiments done.")
    all_bests = [best_23a, m23b, m23c, best_23d, best_23e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-23 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")


if __name__ == "__main__":
    run_experiments()
