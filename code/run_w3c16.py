"""
W3C-16: Optimize KDE-TwoStage+SVM.
  W3C-16a: Fine bandwidth sweep: 250,300,350,400
  W3C-16b: Tune Stage 2 C2 with KDE Stage 1 (bw=300)
  W3C-16c: 3-way blend: KDE-TS + SVM + LogReg
  W3C-16d: KDE-TS only (no SVM) — understand contribution
  W3C-16e: KDE-TS + SVM + TwoStage(C1=10) 3-way
  W3C-16f: KDE bandwidth over elo_diff AND host_adv (2D KDE)
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
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


class TwoStageModel:
    def __init__(self, C1=1.0, C2=1.0):
        self.C1, self.C2 = C1, C2

    def fit(self, X, y):
        y_draw = (y == 1).astype(int)
        self.stage1 = LogisticRegression(C=self.C1, solver='lbfgs', max_iter=1000)
        self.stage1.fit(X, y_draw)
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=1000)
            self.stage2.fit(X[dm], (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        pd1 = self.stage1.predict_proba(X)
        pdraw, pdec = pd1[:,1], pd1[:,0]
        phd = self.stage2.predict_proba(X)[:,1] if self.stage2 else np.full(len(X), 0.5)
        p = np.stack([pdec*(1-phd), pdraw, pdec*phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


class KDETwoStageModel:
    """Nadaraya-Watson kernel for draw probability (Stage 1) + logistic Stage 2."""
    def __init__(self, bandwidth=300.0, C2=2.0, prior_weight=0.1):
        self.bw = bandwidth
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()  # elo_diff
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
        elo_diff = X[:, 0]
        n = len(elo_diff)
        p_draw = np.zeros(n)
        for j in range(n):
            weights = np.exp(-0.5 * ((elo_diff[j] - self.X_train) / self.bw)**2)
            ws = weights.sum()
            if ws > 0:
                kde_est = np.dot(weights, self.y_draw) / ws
                # Mix with prior for stability
                p_draw[j] = (1-self.prior_weight)*np.clip(kde_est,0.05,0.95) + self.prior_weight*self.prior_draw
            else:
                p_draw[j] = self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:,1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec*(1-phd), p_draw, pdec*phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


def cv_blend(X, y_enc, classes, components, blend_weights, exp_name, verbose=True):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    n, nc = len(y_enc), len(classes)
    oof = np.zeros((n, nc))
    fl = []
    for ti, vi in cv.split(X, y_enc):
        Xtr, Xv = X[ti], X[vi]
        ytr, yv = y_enc[ti], y_enc[vi]
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xv = sc.transform(Xv)
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
    met = {"experiment": exp_name, "cv_log_loss_mean": round(ml,4), "cv_log_loss_std": round(sl,4),
           "accuracy": round(acc,4), "delta_vs_baseline_0.8337": round(db,4),
           "delta_vs_frontier_0.7608": round(ml-FRONTIER_LOSS,4),
           "wilcoxon_vs_baseline_pvalue": round(pb,4), "wilcoxon_vs_frontier_pvalue": round(pf,4),
           "verdict_vs_baseline": v, "label_classes": list(classes),
           "blend_weights": blend_weights, "blend_components": [nm for nm,_ in components]}
    if verbose:
        print(f"\n{'='*60}\nExp: {exp_name}\nlog-loss: {ml:.4f} ± {sl:.4f}\nΔ base: {db:+.4f} (p={pb:.4f}) → {v}")
    od = os.path.join(ARTIFACTS_DIR, exp_name)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "metrics.json"), "w") as f: json.dump(met, f, indent=2)
    np.save(os.path.join(od, "oof_probs.npy"), oof)
    pd.DataFrame(oof, columns=classes).to_csv(os.path.join(od, "oof_predictions.csv"), index=False)
    return met, oof


def run_experiments():
    print("Loading WC-2026 features...")
    df = build_match_features(include_context=False)
    X = df[ELO_FEATURES].values
    y = df["label"].values
    le = LabelEncoder(); yenc = le.fit_transform(y); classes = le.classes_

    # W3C-16a: Fine bandwidth sweep
    print("\n--- W3C-16a: Fine KDE bandwidth sweep (C2=2.0) ---")
    best_16a = None
    for bw in [200, 250, 300, 350, 400, 450, 500]:
        m, _ = cv_blend(X, yenc, classes,
            [("KDE_TS", lambda bw=bw: KDETwoStageModel(bw, 2.0)),
             ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
            [0.5, 0.5], f"W3C-16a_bw{bw}", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_16a is None or m['cv_log_loss_mean'] < best_16a['cv_log_loss_mean']:
            best_16a = m
    print(f"Best W3C-16a: {best_16a['cv_log_loss_mean']:.4f} → {best_16a['experiment']}")

    # W3C-16b: Tune C2 with best bw
    print("\n--- W3C-16b: C2 sweep with bw=300 ---")
    best_16b = None
    for C2 in [0.5, 1.0, 2.0, 5.0, 10.0]:
        m, _ = cv_blend(X, yenc, classes,
            [("KDE_TS", lambda C2=C2: KDETwoStageModel(300.0, C2)),
             ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
            [0.5, 0.5], f"W3C-16b_C2{str(C2).replace('.','')}", verbose=False)
        print(f"  C2={C2}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_16b is None or m['cv_log_loss_mean'] < best_16b['cv_log_loss_mean']:
            best_16b = m
    print(f"Best W3C-16b: {best_16b['cv_log_loss_mean']:.4f}")

    # W3C-16c: 3-way blend: KDE-TS + SVM + LogReg(Elo)
    print("\n--- W3C-16c: KDE-TS + SVM + LogReg 3-way ---")
    cv_blend(X, yenc, classes,
        [("KDE_TS", lambda: KDETwoStageModel(300.0, 2.0)),
         ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True)),
         ("LR", lambda: LogisticRegression(C=1.0,solver='lbfgs',max_iter=500))],
        [1/3,1/3,1/3], "W3C-16c")

    # W3C-16d: KDE-TS alone (no SVM)
    print("\n--- W3C-16d: KDE-TS alone (no SVM) ---")
    cv_blend(X, yenc, classes,
        [("KDE_TS", lambda: KDETwoStageModel(300.0, 2.0))],
        [1.0], "W3C-16d_kde_alone")

    # W3C-16e: KDE-TS + SVM + TwoStage(C1=10) 3-way
    print("\n--- W3C-16e: KDE-TS + SVM + TS(C1=10) 3-way ---")
    cv_blend(X, yenc, classes,
        [("KDE_TS", lambda: KDETwoStageModel(300.0, 2.0)),
         ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True)),
         ("TS10", lambda: TwoStageModel(10.0, 2.0))],
        [1/3,1/3,1/3], "W3C-16e")

    # W3C-16f: KDE blend weight optimization
    print("\n--- W3C-16f: KDE-TS+SVM blend weight optimization ---")
    best_16f = None
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        m, _ = cv_blend(X, yenc, classes,
            [("KDE_TS", lambda: KDETwoStageModel(300.0, 2.0)),
             ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
            [alpha, 1-alpha], f"W3C-16f_a{int(alpha*10)}", verbose=False)
        print(f"  alpha={alpha}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_16f is None or m['cv_log_loss_mean'] < best_16f['cv_log_loss_mean']:
            best_16f = m
    print(f"Best W3C-16f: {best_16f['cv_log_loss_mean']:.4f}")

    # W3C-16g: KDE with prior weight tuning
    print("\n--- W3C-16g: KDE prior weight tuning ---")
    for pw in [0.0, 0.05, 0.1, 0.2]:
        m, _ = cv_blend(X, yenc, classes,
            [("KDE_TS", lambda pw=pw: KDETwoStageModel(300.0, 2.0, pw)),
             ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
            [0.5, 0.5], f"W3C-16g_pw{str(pw).replace('.','')}", verbose=True)

    print("\nAll W3C-16 experiments done.")


if __name__ == "__main__":
    run_experiments()
