"""
W3C-12: Build on W3C-11a best (TwoStage(C=2)+SVM).
Experiments:
  W3C-12a: TwoStage(C=2)+SVM+LogReg 3-way blend
  W3C-12b: TwoStage(C=2)+SVM(C=2) 2-way blend
  W3C-12c: TwoStage(C=2)+SVM(C=2)+LogReg 3-way
  W3C-12d: TwoStage(C=2,C2_sweep) - tune Stage 2 independently
  W3C-12e: SVM polynomial kernel + TwoStage(C=2)
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
        self.C1 = C1
        self.C2 = C2

    def fit(self, X, y):
        y_draw = (y == 1).astype(int)
        self.stage1 = LogisticRegression(C=self.C1, solver='lbfgs', max_iter=500)
        self.stage1.fit(X, y_draw)
        decisive_mask = (y != 1)
        if decisive_mask.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=500)
            self.stage2.fit(X[decisive_mask], (y[decisive_mask] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        p_d1 = self.stage1.predict_proba(X)
        p_draw = p_d1[:, 1]
        p_decisive = p_d1[:, 0]
        p_h_dec = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(len(X), 0.5)
        probs = np.stack([p_decisive*(1-p_h_dec), p_draw, p_decisive*p_h_dec], axis=1)
        return probs / probs.sum(axis=1, keepdims=True)


def cv_blend(X, y_enc, classes, components, blend_weights, exp_name, verbose=True):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    n, nc = len(y_enc), len(classes)
    oof_probs = np.zeros((n, nc))
    fold_losses = []
    for train_idx, val_idx in cv.split(X, y_enc):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]
        sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_val_s = sc.transform(X_val)
        bp = np.zeros((len(val_idx), nc))
        for (name, mf), w in zip(components, blend_weights):
            m = mf(); m.fit(X_tr_s, y_tr); bp += w * m.predict_proba(X_val_s)
        oof_probs[val_idx] += bp / N_REPEATS
        fold_losses.append(log_loss(y_val, bp))
    ra = np.array(fold_losses).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    mean_ll, std_ll = ra.mean(), ra.std()
    pm = [-np.log(oof_probs[i, y_enc[i]] + 1e-15) for i in range(n)]
    try: _, p_base = wilcoxon(np.array(pm) - BASELINE_LOSS, alternative='less')
    except: p_base = 1.0
    try: _, p_front = wilcoxon(np.array(pm) - FRONTIER_LOSS, alternative='less')
    except: p_front = 1.0
    delta_base = mean_ll - BASELINE_LOSS
    verdict = "GREEN" if delta_base < -0.01 and p_base < 0.05 else ("RED" if delta_base > 0.01 else "FLAT")
    acc = accuracy_score(y_enc, np.argmax(oof_probs, axis=1))
    metrics = {"experiment": exp_name, "cv_log_loss_mean": round(mean_ll,4),
               "cv_log_loss_std": round(std_ll,4), "accuracy": round(acc,4),
               "delta_vs_baseline_0.8337": round(delta_base,4),
               "delta_vs_frontier_0.7608": round(mean_ll-FRONTIER_LOSS,4),
               "wilcoxon_vs_baseline_pvalue": round(p_base,4),
               "wilcoxon_vs_frontier_pvalue": round(p_front,4),
               "verdict_vs_baseline": verdict, "label_classes": list(classes),
               "blend_weights": blend_weights, "blend_components": [n for n,_ in components]}
    if verbose:
        print(f"\n{'='*60}\nExp: {exp_name}\nlog-loss: {mean_ll:.4f} ± {std_ll:.4f}\n"
              f"Δ base: {delta_base:+.4f} (p={p_base:.4f}) → {verdict}")
    out_dir = os.path.join(ARTIFACTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
    np.save(os.path.join(out_dir, "oof_probs.npy"), oof_probs)
    pd.DataFrame(oof_probs, columns=classes).to_csv(os.path.join(out_dir, "oof_predictions.csv"), index=False)
    return metrics, oof_probs


def run_experiments():
    print("Loading WC-2026 features...")
    df = build_match_features(include_context=False)
    X = df[ELO_FEATURES].values
    y = df["label"].values
    le = LabelEncoder(); y_enc = le.fit_transform(y); classes = le.classes_
    print(f"n={len(y)}, classes={classes}")

    # W3C-12a: TwoStage(C=2)+SVM+LogReg (3-way, equal)
    print("\n--- W3C-12a: TwoStage(C=2)+SVM(C=1)+LogReg, equal 1/3 ---")
    cv_blend(X, y_enc, classes,
        [("TwoStage(C=2)", lambda: TwoStageModel(2.0,2.0)),
         ("SVM-RBF(C=1)", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True)),
         ("LogReg(C=1)", lambda: LogisticRegression(C=1.0,solver='lbfgs',max_iter=500))],
        [1/3,1/3,1/3], "W3C-12a")

    # W3C-12b: TwoStage(C=2)+SVM(C=2) tuned both
    print("\n--- W3C-12b: TwoStage(C=2)+SVM(C=2), equal 50/50 ---")
    cv_blend(X, y_enc, classes,
        [("TwoStage(C=2)", lambda: TwoStageModel(2.0,2.0)),
         ("SVM-RBF(C=2)", lambda: SVC(C=2.0,gamma='scale',kernel='rbf',probability=True))],
        [0.5,0.5], "W3C-12b")

    # W3C-12c: TwoStage(C=2)+SVM(C=2)+LogReg
    print("\n--- W3C-12c: TwoStage(C=2)+SVM(C=2)+LogReg, equal 1/3 ---")
    cv_blend(X, y_enc, classes,
        [("TwoStage(C=2)", lambda: TwoStageModel(2.0,2.0)),
         ("SVM-RBF(C=2)", lambda: SVC(C=2.0,gamma='scale',kernel='rbf',probability=True)),
         ("LogReg(C=1)", lambda: LogisticRegression(C=1.0,solver='lbfgs',max_iter=500))],
        [1/3,1/3,1/3], "W3C-12c")

    # W3C-12d: TwoStage C1/C2 independent sweep (Stage1=draw, Stage2=H/A)
    print("\n--- W3C-12d: TwoStage C1/C2 independent sweep + SVM ---")
    best_12d = None
    for C1 in [1.0, 2.0, 3.0]:
        for C2 in [0.5, 1.0, 2.0]:
            label = f"C1{str(C1).replace('.','')}_C2{str(C2).replace('.','')}"
            m, _ = cv_blend(X, y_enc, classes,
                [("TwoStage", lambda C1=C1,C2=C2: TwoStageModel(C1,C2)),
                 ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
                [0.5,0.5], f"W3C-12d_{label}", verbose=False)
            if best_12d is None or m['cv_log_loss_mean'] < best_12d['cv_log_loss_mean']:
                best_12d = m
    print(f"Best W3C-12d: {best_12d['cv_log_loss_mean']:.4f} (C={best_12d['blend_components']})")
    print(f"  Δ={best_12d['delta_vs_baseline_0.8337']:+.4f}, p={best_12d['wilcoxon_vs_baseline_pvalue']:.4f}, {best_12d['verdict_vs_baseline']}")

    # W3C-12e: Poly kernel SVM + TwoStage(C=2)
    print("\n--- W3C-12e: SVM(poly,degree=2)+TwoStage(C=2) ---")
    cv_blend(X, y_enc, classes,
        [("SVM-poly(d=2)", lambda: SVC(C=1.0,kernel='poly',degree=2,probability=True)),
         ("TwoStage(C=2)", lambda: TwoStageModel(2.0,2.0))],
        [0.5,0.5], "W3C-12e")

    print("\n--- W3C-12f: SVM(poly,degree=3)+TwoStage(C=2) ---")
    cv_blend(X, y_enc, classes,
        [("SVM-poly(d=3)", lambda: SVC(C=1.0,kernel='poly',degree=3,probability=True)),
         ("TwoStage(C=2)", lambda: TwoStageModel(2.0,2.0))],
        [0.5,0.5], "W3C-12f")

    print("\nAll W3C-12 experiments done.")


if __name__ == "__main__":
    run_experiments()
