"""
W3C-29: Feature expansion for Stage 2 (H/A discrimination).
New discovery: mv_top11_diff corr=0.585, rank_diff corr=0.531, gk_mv_diff corr=0.504
alongside elo_diff corr=0.587. These may improve Stage 2 beyond elo+host.

  a) Stage 2 feature expansion: add rank_diff, mv_top11_diff to Stage 2
  b) Alternative Stage 1 KDE feature: mv_top11_diff instead of elo_diff
  c) 2D KDE (elo_diff, mv_top11_diff) for Stage 1
  d) mv_top11_diff + SVM blend (independent model)
  e) Ensemble: original KDE-TS + mv_top11 KDE-TS + SVM
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

# Extended feature sets
STAGE2_FEATURES_BASE = ['elo_diff', 'host_advantage']  # same as ELO_FEATURES
STAGE2_FEATURES_EXT = ['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff']


class KDETSFlexModel:
    """KDE Stage 1 (on one feature) + LogReg Stage 2 (on a separate feature set)."""
    def __init__(self, stage1_col=0, bw=300.0, C2=2.0, prior_weight=0.2, stage2_cols=None):
        self.stage1_col = stage1_col
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight
        self.stage2_cols = stage2_cols  # if None, uses all features

    def fit(self, X, y):
        self.X_train = X[:, self.stage1_col].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        X_dec = X[dm] if self.stage2_cols is None else X[np.ix_(dm, self.stage2_cols)]
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=1000)
            self.stage2.fit(X_dec, (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        elo, n = X[:, self.stage1_col], len(X)
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
        X_dec = X if self.stage2_cols is None else X[:, self.stage2_cols]
        phd = self.stage2.predict_proba(X_dec)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


class KDE2DDrawModel:
    """2D KDE for Stage 1 draw probability: uses elo_diff and mv_top11_diff."""
    def __init__(self, bw0=300.0, bw1=1.0, C2=2.0, prior_weight=0.2, cols=(0, 2)):
        self.bw0 = bw0; self.bw1 = bw1; self.C2 = C2
        self.prior_weight = prior_weight; self.cols = cols

    def fit(self, X, y):
        self.X_train = X[:, list(self.cols)].copy()
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
        X2 = X[:, list(self.cols)]
        n = len(X)
        p_draw = np.zeros(n)
        for j in range(n):
            d0 = (X2[j, 0] - self.X_train[:, 0]) / self.bw0
            d1 = (X2[j, 1] - self.X_train[:, 1]) / self.bw1
            w = np.exp(-0.5 * (d0**2 + d1**2))
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
    print("Loading WC-2026 features (extended)...")
    df = build_match_features(include_context=False)
    # Extended feature matrix
    FEAT_EXT = ['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff']
    X_base = df[ELO_FEATURES].values          # [elo_diff, host_advantage]
    X_ext = df[FEAT_EXT].values               # 5 features
    X_mv = df[['mv_top11_diff', 'host_advantage']].values   # mv-based 2-feature set
    y = df["label"].values
    le = LabelEncoder()
    yenc = le.fit_transform(y)
    classes = le.classes_
    print(f"n={len(yenc)}, classes={classes}")
    print(f"X_ext features: {FEAT_EXT}")

    svm_fn = lambda: SVC(C=0.35, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    svm_c1 = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED)

    # ── W3C-29a: Stage 2 uses extended features (rank_diff, mv_top11_diff added) ─
    print("\n--- W3C-29a: Stage 2 with extended features (5-feat set) ---")
    best_29a = None
    # stage2_cols indexes into the X_ext feature matrix
    # elo_diff=0, host_advantage=1, rank_diff=2, mv_top11_diff=3, gk_mv_diff=4
    for s2_cols, tag in [
        ([0, 1], "base"),
        ([0, 1, 2], "base_rank"),
        ([0, 1, 3], "base_mv"),
        ([0, 1, 4], "base_gk"),
        ([0, 1, 2, 3], "base_rank_mv"),
        ([0, 1, 2, 3, 4], "all5"),
        ([0, 1, 3, 4], "base_mv_gk"),
    ]:
        # Stage 1 always uses col 0 (elo_diff); Stage 2 uses s2_cols
        kde_fn = lambda s2c=s2_cols: KDETSFlexModel(stage1_col=0, bw=300.0, C2=2.0,
                                                    prior_weight=0.2, stage2_cols=s2c)
        m = cv_blend(X_ext, yenc, classes,
            [("KDE_TS_ext", kde_fn), ("SVM", svm_fn)],
            [0.70, 0.30], f"W3C-29a_s2_{tag}", verbose=False)
        print(f"  Stage2 cols={[FEAT_EXT[c] for c in s2_cols]}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        if best_29a is None or m['cv_log_loss_mean'] < best_29a['cv_log_loss_mean']:
            best_29a = m
    print(f"Best W3C-29a: {best_29a['cv_log_loss_mean']:.4f} → {best_29a['experiment']}")

    # ── W3C-29b: Stage 1 KDE on mv_top11_diff (col 3 in X_ext) ─────────────
    print("\n--- W3C-29b: Stage 1 KDE on mv_top11_diff + SVM ---")
    best_29b = None
    for bw in [0.5, 1.0, 1.5, 2.0, 3.0]:
        kde_mv = lambda bw=bw: KDETSFlexModel(stage1_col=3, bw=bw, C2=2.0, prior_weight=0.2)
        m = cv_blend(X_ext, yenc, classes,
            [("KDE_mv", kde_mv), ("SVM", svm_c1)],
            [0.70, 0.30], f"W3C-29b_mv_bw{int(bw*10)}", verbose=False)
        print(f"  mv_bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_29b is None or m['cv_log_loss_mean'] < best_29b['cv_log_loss_mean']:
            best_29b = m
    print(f"Best W3C-29b: {best_29b['cv_log_loss_mean']:.4f}")

    # ── W3C-29c: 2D KDE (elo_diff, mv_top11_diff) + SVM ─────────────────────
    print("\n--- W3C-29c: 2D KDE (elo_diff, mv_top11_diff) + SVM ---")
    best_29c = None
    for bw1 in [0.3, 0.5, 1.0, 2.0]:
        kde_2d = lambda bw1=bw1: KDE2DDrawModel(bw0=300.0, bw1=bw1, C2=2.0, prior_weight=0.2, cols=(0, 3))
        m = cv_blend(X_ext, yenc, classes,
            [("KDE_2D", kde_2d), ("SVM", svm_fn)],
            [0.70, 0.30], f"W3C-29c_2D_bw1_{int(bw1*10)}", verbose=False)
        print(f"  bw_mv={bw1}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_29c is None or m['cv_log_loss_mean'] < best_29c['cv_log_loss_mean']:
            best_29c = m
    print(f"Best W3C-29c: {best_29c['cv_log_loss_mean']:.4f}")

    # ── W3C-29d: mv_top11_diff KDE-TS + SVM blend ────────────────────────────
    print("\n--- W3C-29d: mv_top11_diff KDE-TS + SVM (alpha sweep) ---")
    best_29d = None
    kde_mv_best = lambda: KDETSFlexModel(stage1_col=3, bw=1.0, C2=2.0, prior_weight=0.2)
    for alpha in [0.5, 0.6, 0.7, 0.8]:
        m = cv_blend(X_ext, yenc, classes,
            [("KDE_mv", kde_mv_best), ("SVM", svm_c1)],
            [alpha, 1 - alpha], f"W3C-29d_mv_a{int(alpha*10)}", verbose=False)
        print(f"  alpha={alpha}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_29d is None or m['cv_log_loss_mean'] < best_29d['cv_log_loss_mean']:
            best_29d = m
    print(f"Best W3C-29d: {best_29d['cv_log_loss_mean']:.4f}")

    # ── W3C-29e: 3-component: elo KDE-TS + mv KDE-TS + SVM ──────────────────
    print("\n--- W3C-29e: 3-component blend: elo KDE + mv KDE + SVM ---")
    kde_elo_fn = lambda: KDETSFlexModel(stage1_col=0, bw=300.0, C2=2.0, prior_weight=0.2)
    kde_mv_fn = lambda: KDETSFlexModel(stage1_col=3, bw=1.0, C2=2.0, prior_weight=0.2)
    best_29e = None
    for we, wm, ws in [
        (0.45, 0.25, 0.30),
        (0.40, 0.30, 0.30),
        (0.50, 0.20, 0.30),
        (0.35, 0.35, 0.30),
        (0.40, 0.25, 0.35),
        (0.50, 0.25, 0.25),
    ]:
        tag = f"e{int(we*100)}m{int(wm*100)}s{int(ws*100)}"
        m = cv_blend(X_ext, yenc, classes,
            [("EloKDE", kde_elo_fn), ("MvKDE", kde_mv_fn), ("SVM", svm_fn)],
            [we, wm, ws], f"W3C-29e_{tag}", verbose=False)
        print(f"  elo={we} mv={wm} svm={ws}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_29e is None or m['cv_log_loss_mean'] < best_29e['cv_log_loss_mean']:
            best_29e = m
    print(f"Best W3C-29e: {best_29e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-29 experiments done.")
    all_bests = [best_29a, best_29b, best_29c, best_29d, best_29e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-29 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")

    import json
    results = sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])
    with open("/tmp/w3c29_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "note": "Feature expansion: rank_diff, mv_top11_diff, gk_mv_diff added to Stage 2",
                   "all_bests": results}, f, indent=2)
    print("Summary saved to /tmp/w3c29_summary.json")


if __name__ == "__main__":
    run_experiments()
