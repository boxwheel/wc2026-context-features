"""
W3C-13: Extended optimization of TwoStage(C1=3,C2=2)+SVM blend.
  W3C-13a: C1 fine-tuning (3.0, 4.0, 5.0, 7.0, 10.0) with C2=2.0
  W3C-13b: 3-way blend: TwoStage(C1=3,C2=2)+SVM+LogReg
  W3C-13c: OvR (one-vs-rest) binary SVM classifiers, normalized
  W3C-13d: Dirichlet prior blend - add uniform prior to sharpen probabilities
  W3C-13e: LogisticRegression(C=0.5, 1.0, 2.0) - find best Elo LogReg alone
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
        self.stage1 = LogisticRegression(C=self.C1, solver='lbfgs', max_iter=500)
        self.stage1.fit(X, y_draw)
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=500)
            self.stage2.fit(X[dm], (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        pd1 = self.stage1.predict_proba(X)
        pd, pde = pd1[:,1], pd1[:,0]
        phd = self.stage2.predict_proba(X)[:,1] if self.stage2 else np.full(len(X), 0.5)
        p = np.stack([pde*(1-phd), pd, pde*phd], axis=1)
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
           "blend_weights": blend_weights, "blend_components": [n for n,_ in components]}
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
    print(f"n={len(y)}, classes={classes}")

    # W3C-13a: TwoStage C1 fine-tune (C2=2.0 fixed)
    print("\n--- W3C-13a: TwoStage C1 fine-tune (C2=2.0 fixed) + SVM ---")
    best_13a = None
    for C1 in [2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        lbl = str(C1).replace('.','')
        m, _ = cv_blend(X, yenc, classes,
            [("TS", lambda C1=C1: TwoStageModel(C1, 2.0)),
             ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
            [0.5, 0.5], f"W3C-13a_C1{lbl}", verbose=True)
        if best_13a is None or m['cv_log_loss_mean'] < best_13a['cv_log_loss_mean']:
            best_13a = m
    print(f"Best W3C-13a: {best_13a['cv_log_loss_mean']:.4f}")

    # W3C-13b: 3-way blend TwoStage(C1=3,C2=2)+SVM+LogReg
    print("\n--- W3C-13b: TwoStage(C1=3,C2=2)+SVM+LogReg, 1/3 each ---")
    cv_blend(X, yenc, classes,
        [("TS(3,2)", lambda: TwoStageModel(3.0, 2.0)),
         ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True)),
         ("LR", lambda: LogisticRegression(C=1.0,solver='lbfgs',max_iter=500))],
        [1/3,1/3,1/3], "W3C-13b")

    # W3C-13c: LogReg C-sweep to find best standalone LogReg
    print("\n--- W3C-13c: LogReg C-sweep (standalone, just Elo) ---")
    for C in [0.5, 1.0, 2.0, 5.0]:
        cv_blend(X, yenc, classes,
            [("LR", lambda C=C: LogisticRegression(C=C,solver='lbfgs',max_iter=500))],
            [1.0], f"W3C-13c_C{str(C).replace('.','')}")

    # W3C-13d: 4-way blend: TS(3,2)+TS(1,2)+SVM+LR
    print("\n--- W3C-13d: TS(3,2)+TS(1,2)+SVM+LR, 1/4 each ---")
    cv_blend(X, yenc, classes,
        [("TS(3,2)", lambda: TwoStageModel(3.0, 2.0)),
         ("TS(1,2)", lambda: TwoStageModel(1.0, 2.0)),
         ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True)),
         ("LR", lambda: LogisticRegression(C=1.0,solver='lbfgs',max_iter=500))],
        [0.25,0.25,0.25,0.25], "W3C-13d")

    # W3C-13e: TS(3,2)+SVM with SVM C-sweep
    print("\n--- W3C-13e: TS(3,2)+SVM(C sweep) blend ---")
    for C_svm in [0.5, 1.5, 2.0, 3.0]:
        cv_blend(X, yenc, classes,
            [("TS(3,2)", lambda: TwoStageModel(3.0, 2.0)),
             ("SVM", lambda C=C_svm: SVC(C=C,gamma='scale',kernel='rbf',probability=True))],
            [0.5, 0.5], f"W3C-13e_svm{str(C_svm).replace('.','')}")

    print("\nAll W3C-13 experiments done.")


if __name__ == "__main__":
    run_experiments()
