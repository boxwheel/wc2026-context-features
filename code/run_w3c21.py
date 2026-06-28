"""
W3C-21: New ensemble components — Random Forest, ExtraTrees, GBM.
  a) KDE-TS + RandomForest (max_depth sweep)
  b) KDE-TS + ExtraTreesClassifier (extreme randomization)
  c) KDE-TS + GBM (max_depth=1 stumps, lr/n_est sweep)
  d) KDE-TS + SVM + RF three-way blend
  e) KDE-TS + SVM + GBM three-way blend
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier)
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

    # ── W3C-21a: KDE-TS + RandomForest (max_depth sweep) ────────────────────
    print("\n--- W3C-21a: KDE-TS + RandomForest (max_depth sweep) ---")
    best_21a = None
    for md in [3, 5, 7, None]:
        for ne in [200, 500]:
            tag = f"md{md or 'none'}_ne{ne}"
            rf_fn = lambda md=md, ne=ne: RandomForestClassifier(
                n_estimators=ne, max_depth=md, random_state=SEED, n_jobs=-1)
            m = cv_blend(X, yenc, classes,
                [("KDE_TS", kde_best), ("RF", rf_fn)],
                [0.5, 0.5], f"W3C-21a_rf_{tag}", verbose=False)
            print(f"  RF(max_depth={md}, n={ne}): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
            if best_21a is None or m['cv_log_loss_mean'] < best_21a['cv_log_loss_mean']:
                best_21a = m
    print(f"Best W3C-21a: {best_21a['cv_log_loss_mean']:.4f} → {best_21a['experiment']}")

    # ── W3C-21b: KDE-TS + ExtraTrees ────────────────────────────────────────
    print("\n--- W3C-21b: KDE-TS + ExtraTreesClassifier ---")
    best_21b = None
    for md in [5, 7, None]:
        et_fn = lambda md=md: ExtraTreesClassifier(
            n_estimators=500, max_depth=md, random_state=SEED, n_jobs=-1)
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best), ("ET", et_fn)],
            [0.5, 0.5], f"W3C-21b_et_md{md or 'none'}", verbose=False)
        print(f"  ET(max_depth={md}): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_21b is None or m['cv_log_loss_mean'] < best_21b['cv_log_loss_mean']:
            best_21b = m
    print(f"Best W3C-21b: {best_21b['cv_log_loss_mean']:.4f}")

    # ── W3C-21c: KDE-TS + GBM (stumps, sweep lr+n_estimators) ──────────────
    print("\n--- W3C-21c: KDE-TS + GBM (stump, lr/n_est sweep) ---")
    best_21c = None
    for lr, ne in [(0.05, 200), (0.1, 100), (0.1, 200), (0.05, 500), (0.01, 1000)]:
        gbm_fn = lambda lr=lr, ne=ne: GradientBoostingClassifier(
            n_estimators=ne, learning_rate=lr, max_depth=1,
            subsample=0.8, random_state=SEED)
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best), ("GBM", gbm_fn)],
            [0.5, 0.5], f"W3C-21c_gbm_lr{int(lr*100)}_ne{ne}", verbose=False)
        print(f"  GBM(lr={lr}, n={ne}): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_21c is None or m['cv_log_loss_mean'] < best_21c['cv_log_loss_mean']:
            best_21c = m
    print(f"Best W3C-21c: {best_21c['cv_log_loss_mean']:.4f}")

    # ── W3C-21d: Best RF alpha sweep (KDE weight) ───────────────────────────
    print("\n--- W3C-21d: Best RF — alpha sweep ---")
    # Extract best RF params from 21a
    best_21d = None
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        rf_fn = lambda: RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1)
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best), ("RF", rf_fn)],
            [alpha, 1 - alpha], f"W3C-21d_rf_a{int(alpha*10)}", verbose=False)
        print(f"  alpha(KDE)={alpha}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_21d is None or m['cv_log_loss_mean'] < best_21d['cv_log_loss_mean']:
            best_21d = m
    print(f"Best W3C-21d: {best_21d['cv_log_loss_mean']:.4f}")

    # ── W3C-21e: KDE-TS + SVM + RF three-way ────────────────────────────────
    print("\n--- W3C-21e: KDE-TS + SVM + RF three-way ---")
    best_21e = None
    for w_kde, w_svm, w_rf in [(1/3, 1/3, 1/3), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3), (0.5, 0.3, 0.2)]:
        rf_fn = lambda: RandomForestClassifier(n_estimators=500, max_depth=5, random_state=SEED, n_jobs=-1)
        tag = f"k{int(w_kde*10)}_s{int(w_svm*10)}_r{int(w_rf*10)}"
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best), ("SVM", svm_fn), ("RF", rf_fn)],
            [w_kde, w_svm, w_rf], f"W3C-21e_3way_{tag}", verbose=False)
        print(f"  (KDE={w_kde:.1f}, SVM={w_svm:.1f}, RF={w_rf:.1f}): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_21e is None or m['cv_log_loss_mean'] < best_21e['cv_log_loss_mean']:
            best_21e = m
    print(f"Best W3C-21e: {best_21e['cv_log_loss_mean']:.4f}")

    # ── W3C-21f: KDE-TS + SVM + GBM three-way ───────────────────────────────
    print("\n--- W3C-21f: KDE-TS + SVM + GBM three-way ---")
    best_21f = None
    gbm_best = lambda: GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=1, subsample=0.8, random_state=SEED)
    for w_kde, w_svm, w_gbm in [(1/3, 1/3, 1/3), (0.4, 0.4, 0.2), (0.4, 0.3, 0.3), (0.5, 0.3, 0.2)]:
        tag = f"k{int(w_kde*10)}_s{int(w_svm*10)}_g{int(w_gbm*10)}"
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best), ("SVM", svm_fn), ("GBM", gbm_best)],
            [w_kde, w_svm, w_gbm], f"W3C-21f_3way_{tag}", verbose=False)
        print(f"  (KDE={w_kde:.1f}, SVM={w_svm:.1f}, GBM={w_gbm:.1f}): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_21f is None or m['cv_log_loss_mean'] < best_21f['cv_log_loss_mean']:
            best_21f = m
    print(f"Best W3C-21f: {best_21f['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-21 experiments done.")
    all_bests = [best_21a, best_21b, best_21c, best_21d, best_21e, best_21f]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-21 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")


if __name__ == "__main__":
    run_experiments()
