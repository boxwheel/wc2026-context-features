"""
W3C-32: Ultra-fine grid + alternative model components.
W3C-31 found alpha=0.45, C=2.0 → 0.7745 GREEN as robust optimum.
  a) Ultra-fine (alpha, C) grid around (0.45, 2.0)
  b) Random Forest(max_depth=3) on 5 features at alpha=0.45
  c) GradientBoosting on 5 features at alpha=0.45
  d) Multinomial LogReg (direct multi-class) on 5 features at alpha=0.45
  e) KNN on 5 features at alpha=0.45 (k sweep)
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
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
        self.bw = bw; self.C2 = C2; self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=1000)
            self.stage2.fit(X[dm, :2], (y[dm] == 2).astype(int))
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
        phd = self.stage2.predict_proba(X[:, :2])[:, 1] if self.stage2 else np.full(n, 0.5)
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
    y = df["label"].values
    le = LabelEncoder()
    yenc = le.fit_transform(y)
    classes = le.classes_
    print(f"n={len(yenc)}, classes={classes}")

    FEAT5 = ['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff']
    X5 = df[FEAT5].values
    kde_fn = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)

    # ── W3C-32a: Ultra-fine (alpha, C) grid around (0.45, 2.0) ──────────────
    print("\n--- W3C-32a: Ultra-fine grid alpha x C around (0.45, 2.0) ---")
    best_32a = None
    for alpha in [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52]:
        for C in [1.5, 1.7, 2.0, 2.3, 2.7]:
            svm_c = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            tag = f"a{int(alpha*100)}_C{int(C*10):02d}"
            m = cv_blend(X5, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM5", svm_c)],
                [alpha, 1 - alpha], f"W3C-32a_{tag}", verbose=False)
            if best_32a is None or m['cv_log_loss_mean'] < best_32a['cv_log_loss_mean']:
                best_32a = m
        # Print best for this alpha
        best_row = [x for x in [best_32a] if True]
        row_results = [(alpha, C) for C in [1.5, 1.7, 2.0, 2.3, 2.7]]
    # Print progress
    print(f"Best so far: {best_32a['cv_log_loss_mean']:.4f} → {best_32a['experiment']}")

    # ── W3C-32b: Random Forest on 5 features at alpha=0.45 ───────────────────
    print("\n--- W3C-32b: RandomForest(5-feat) at alpha=0.45 ---")
    best_32b = None
    for depth, ne in [(3, 100), (3, 200), (3, 500), (4, 100), (4, 200), (5, 100)]:
        rf_fn = lambda d=depth, n=ne: RandomForestClassifier(max_depth=d, n_estimators=n, random_state=SEED, n_jobs=-1)
        tag = f"d{depth}n{ne}"
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("RF5", rf_fn)],
            [0.45, 0.55], f"W3C-32b_{tag}", verbose=False)
        print(f"  RF5 d={depth} n={ne}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        if best_32b is None or m['cv_log_loss_mean'] < best_32b['cv_log_loss_mean']:
            best_32b = m
    print(f"Best W3C-32b: {best_32b['cv_log_loss_mean']:.4f}")

    # ── W3C-32c: GradientBoosting on 5 features at alpha=0.45 ────────────────
    print("\n--- W3C-32c: GradientBoosting(5-feat) at alpha=0.45 ---")
    best_32c = None
    for lr2, depth in [(0.1, 1), (0.05, 1), (0.1, 2), (0.05, 2)]:
        gbm_fn = lambda lr=lr2, d=depth: GradientBoostingClassifier(
            learning_rate=lr, max_depth=d, n_estimators=50, random_state=SEED)
        tag = f"lr{int(lr2*100)}_d{depth}"
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("GBM5", gbm_fn)],
            [0.45, 0.55], f"W3C-32c_{tag}", verbose=False)
        print(f"  GBM lr={lr2} d={depth}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_32c is None or m['cv_log_loss_mean'] < best_32c['cv_log_loss_mean']:
            best_32c = m
    print(f"Best W3C-32c: {best_32c['cv_log_loss_mean']:.4f}")

    # ── W3C-32d: Multinomial LR on 5 features ────────────────────────────────
    print("\n--- W3C-32d: Multinomial LogReg(5-feat) at alpha=0.45 ---")
    best_32d = None
    for C, alpha in [(0.1, 0.45), (0.3, 0.45), (0.5, 0.45), (1.0, 0.45), (2.0, 0.45),
                     (1.0, 0.40), (1.0, 0.50), (2.0, 0.40), (2.0, 0.50)]:
        lr_fn = lambda C=C: LogisticRegression(C=C, solver='lbfgs', max_iter=1000,
                                               multi_class='multinomial', random_state=SEED)
        tag = f"C{int(C*10):02d}_a{int(alpha*100)}"
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("MLR", lr_fn)],
            [alpha, 1 - alpha], f"W3C-32d_{tag}", verbose=False)
        print(f"  MLR C={C} alpha={alpha}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_32d is None or m['cv_log_loss_mean'] < best_32d['cv_log_loss_mean']:
            best_32d = m
    print(f"Best W3C-32d: {best_32d['cv_log_loss_mean']:.4f}")

    # ── W3C-32e: KNN on 5 features at alpha=0.45 ─────────────────────────────
    print("\n--- W3C-32e: KNN(5-feat) at alpha=0.45 ---")
    best_32e = None
    for k in [3, 5, 7, 9, 11, 15, 20]:
        knn_fn = lambda k=k: KNeighborsClassifier(n_neighbors=k, weights='distance')
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("KNN5", knn_fn)],
            [0.45, 0.55], f"W3C-32e_k{k}", verbose=False)
        print(f"  KNN k={k}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_32e is None or m['cv_log_loss_mean'] < best_32e['cv_log_loss_mean']:
            best_32e = m
    print(f"Best W3C-32e: {best_32e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-32 experiments done.")
    all_bests = [best_32a, best_32b, best_32c, best_32d, best_32e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-32 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")

    import json
    results = sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])
    with open("/tmp/w3c32_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "note": "Ultra-fine grid + alternative model components at alpha=0.45",
                   "all_bests": results}, f, indent=2)
    print("Summary saved to /tmp/w3c32_summary.json")


if __name__ == "__main__":
    run_experiments()
