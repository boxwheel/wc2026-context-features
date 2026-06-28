"""
W3C-32d/e continuation: run only the KNN and multinomial LR parts
after the W3C-32 crash (multi_class deprecated in sklearn 1.5+).
  d) Multinomial LogReg (direct multi-class) on 5 features at alpha=0.45
  e) KNN on 5 features at alpha=0.45 (k sweep)
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from features import build_match_features

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

    # ── W3C-32d: Multinomial LogReg on 5 features (fixed: no multi_class kwarg) ──
    print("\n--- W3C-32d: Multinomial LogReg(5-feat) at alpha=0.45 ---")
    best_32d = None
    for C, alpha in [(0.1, 0.45), (0.3, 0.45), (0.5, 0.45), (1.0, 0.45), (2.0, 0.45),
                     (1.0, 0.40), (1.0, 0.50), (2.0, 0.40), (2.0, 0.50)]:
        # sklearn 1.5+ removed multi_class kwarg; lbfgs defaults to multinomial
        lr_fn = lambda C=C: LogisticRegression(C=C, solver='lbfgs', max_iter=1000,
                                               random_state=SEED)
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

    print("\nW3C-32d/e done.")
    with open("/tmp/w3c32de_summary.json", "w") as f:
        json.dump({"best_32d": best_32d, "best_32e": best_32e}, f, indent=2)
    print("Summary saved to /tmp/w3c32de_summary.json")


if __name__ == "__main__":
    run_experiments()
