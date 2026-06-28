"""
W3C-14: Push C1 higher for TwoStage + SVM. Also explore:
  - LogisticRegression with C=1e6 (essentially unregularized)
  - Adding poly-2 elo_diff^2 feature to the blend
  - Gradient-boosted TwoStage probabilities
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

    # W3C-14a: Higher C1 sweep (C2=2.0 fixed, SVM C=1.0)
    print("\n--- W3C-14a: Higher C1 sweep (C2=2.0 fixed) + SVM ---")
    best_14a = None
    for C1 in [20.0, 50.0, 100.0, 500.0, 1000.0]:
        lbl = str(int(C1))
        m, _ = cv_blend(X, yenc, classes,
            [("TS", lambda C1=C1: TwoStageModel(C1, 2.0)),
             ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
            [0.5, 0.5], f"W3C-14a_C1{lbl}")
        if best_14a is None or m['cv_log_loss_mean'] < best_14a['cv_log_loss_mean']:
            best_14a = m
    print(f"Best W3C-14a: {best_14a['cv_log_loss_mean']:.4f} → {best_14a['experiment']}")

    # W3C-14b: With optimal C1, tune C2 more broadly 
    print("\n--- W3C-14b: Best C1 + C2 sweep (0.5,1,2,5,10,50) ---")
    best_C1 = float(best_14a['experiment'].split('C1')[-1])
    best_14b = None
    for C2 in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        lbl = str(C2).replace('.','')
        m, _ = cv_blend(X, yenc, classes,
            [("TS", lambda C2=C2: TwoStageModel(best_C1, C2)),
             ("SVM", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
            [0.5, 0.5], f"W3C-14b_C2{lbl}", verbose=False)
        if best_14b is None or m['cv_log_loss_mean'] < best_14b['cv_log_loss_mean']:
            best_14b = m
        print(f"  C2={C2}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
    print(f"Best W3C-14b: {best_14b['cv_log_loss_mean']:.4f} → {best_14b['experiment']}")

    # W3C-14c: Add poly-2 feature (elo_diff^2) to the blend
    print("\n--- W3C-14c: TwoStage + SVM with elo_diff, host_adv, elo_diff^2 ---")
    X_poly = np.column_stack([X, X[:,0]**2])  # elo_diff^2
    cv_blend(X_poly, yenc, classes,
        [("TS_poly", lambda: TwoStageModel(10.0, 2.0)),
         ("SVM_poly", lambda: SVC(C=1.0,gamma='scale',kernel='rbf',probability=True))],
        [0.5, 0.5], "W3C-14c_poly")

    # W3C-14d: TwoStage(C1=best, C2=best) alone (no SVM)
    print("\n--- W3C-14d: TwoStage alone (best C1, C2=2) ---")
    best_C2 = float(best_14b['experiment'].split('C2')[-1].replace('0','0.') if '.' not in best_14b['experiment'].split('C2')[-1] else best_14b['experiment'].split('C2')[-1])
    try: best_C2 = float(best_14b['blend_components'][0].split(',')[-1].replace(')',''))
    except: best_C2 = 2.0
    cv_blend(X, yenc, classes,
        [("TS_alone", lambda: TwoStageModel(best_C1, 2.0))],
        [1.0], "W3C-14d_ts_alone")

    print("\nAll W3C-14 experiments done.")


if __name__ == "__main__":
    run_experiments()
