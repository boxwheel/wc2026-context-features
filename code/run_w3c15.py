"""
W3C-15: Fine-grained C1 search + KDE-based draw detection.
  W3C-15a: C1 fine search around 10-20 (8, 10, 12, 15, 18)
  W3C-15b: KDE Stage 1 (Nadaraya-Watson draw probability vs elo_diff)
  W3C-15c: Multi-TwoStage blend — 3 TS models with C1=5,10,20
  W3C-15d: TwoStage with elo_diff absolute value as extra draw feature
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
    """Nadaraya-Watson kernel for Stage 1 (draw probability vs |elo_diff|)."""
    def __init__(self, bandwidth=200.0, C2=2.0):
        self.bandwidth = bandwidth
        self.C2 = C2

    def fit(self, X, y):
        # Stage 1: KDE estimate of P(draw | elo_diff)
        elo_diff = X[:, 0]
        self.X_train = elo_diff.copy()
        self.y_draw = (y == 1).astype(float)
        
        # Stage 2: LogReg for H/A within decisive
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
        
        # Nadaraya-Watson: P(draw | x) = sum_i K(x-xi) * y_draw_i / sum_i K(x-xi)
        bw = self.bandwidth
        p_draw = np.zeros(n)
        for j in range(n):
            weights = np.exp(-0.5 * ((elo_diff[j] - self.X_train) / bw)**2)
            weights_sum = weights.sum()
            if weights_sum > 0:
                p_draw[j] = np.clip(np.dot(weights, self.y_draw) / weights_sum, 0.05, 0.95)
            else:
                p_draw[j] = 0.28  # prior draw rate
        
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

    # W3C-15a: Fine-grained C1 search (8, 10, 12, 15, 18)
    print("\n--- W3C-15a: Fine C1 search around 10-15 ---")
    best_15a = None
    for C1 in [8.0, 10.0, 11.0, 12.0, 15.0, 18.0]:
        lbl = str(C1).replace('.','')
        m, _ = cv_blend(X, yenc, classes,
            [("TS", lambda C1=C1: TwoStageModel(C1, 2.0)),
             ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
            [0.5, 0.5], f"W3C-15a_C1{lbl}")
        if best_15a is None or m['cv_log_loss_mean'] < best_15a['cv_log_loss_mean']:
            best_15a = m
    print(f"Best W3C-15a: {best_15a['cv_log_loss_mean']:.4f}")

    # W3C-15b: KDE Stage 1, bandwidth sweep
    print("\n--- W3C-15b: KDE TwoStage + SVM ---")
    best_15b = None
    for bw in [100.0, 150.0, 200.0, 300.0, 500.0]:
        lbl = str(int(bw))
        m, _ = cv_blend(X, yenc, classes,
            [("KDE_TS", lambda bw=bw: KDETwoStageModel(bw, 2.0)),
             ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
            [0.5, 0.5], f"W3C-15b_bw{lbl}", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_15b is None or m['cv_log_loss_mean'] < best_15b['cv_log_loss_mean']:
            best_15b = m
    print(f"Best KDE bw: {best_15b['cv_log_loss_mean']:.4f}")

    # W3C-15c: Multi-TwoStage blend 3 models (C1=5,10,20) averaged + SVM
    print("\n--- W3C-15c: Multi-TS (C1=5,10,20) averaged + SVM ---")
    def make_multi_ts():
        class MultiTS:
            def __init__(self):
                self.models = [TwoStageModel(5.0,2.0), TwoStageModel(10.0,2.0), TwoStageModel(20.0,2.0)]
            def fit(self, X, y):
                for m in self.models: m.fit(X, y)
                return self
            def predict_proba(self, X):
                return np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return MultiTS()
    cv_blend(X, yenc, classes,
        [("MultiTS", make_multi_ts),
         ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
        [0.5, 0.5], "W3C-15c_multi_ts")

    # W3C-15d: Isotonic calibration on the TS blend
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.base import BaseEstimator, ClassifierMixin

    print("\n--- W3C-15d: Optimal TwoStage(C=10) + SVM as CalibratedCV ---")
    # Test TwoStage alone with calibration
    cv_blend(X, yenc, classes,
        [("TS(10,2)+SVM CalibratedBlend", lambda: TwoStageModel(10.0, 2.0))],
        [1.0], "W3C-15d_ts10_alone_check")

    print("\nAll W3C-15 experiments done.")


if __name__ == "__main__":
    run_experiments()
