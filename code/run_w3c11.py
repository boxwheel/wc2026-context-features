"""
W3C-11: Optimize the 3-component blend (LogReg + SVM + TwoStage).
Experiments:
  W3C-11a: TwoStage parameter sweep (C1, C2) paired with SVM
  W3C-11b: Blend weight optimization for LogReg+SVM+TwoStage
  W3C-11c: LogReg + SVM + TwoStage + TwoStage(C=0.5) (4-model)
  W3C-11d: SVM(C=0.5) + TwoStage - try lower C SVM
  W3C-11e: SVM(C=2.0) + TwoStage - try higher C SVM
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
            X_dec = X[decisive_mask]
            y_dec = (y[decisive_mask] == 2).astype(int)
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=500)
            self.stage2.fit(X_dec, y_dec)
        else:
            self.stage2 = None
        return self

    def predict_proba(self, X):
        p_d1 = self.stage1.predict_proba(X)
        p_draw = p_d1[:, 1]
        p_decisive = p_d1[:, 0]
        if self.stage2 is not None:
            p_h_dec = self.stage2.predict_proba(X)[:, 1]
        else:
            p_h_dec = np.full(len(X), 0.5)
        p_h = p_decisive * p_h_dec
        p_a = p_decisive * (1 - p_h_dec)
        probs = np.stack([p_a, p_draw, p_h], axis=1)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


def cv_blend_evaluate(X, y_enc, classes, components, blend_weights, exp_name, verbose=True):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    n, nc = len(y_enc), len(classes)

    oof_probs = np.zeros((n, nc))
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_enc)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        blend_prob = np.zeros((len(val_idx), nc))
        for (name, model_factory), w in zip(components, blend_weights):
            m = model_factory()
            m.fit(X_tr_s, y_tr)
            blend_prob += w * m.predict_proba(X_val_s)
        oof_probs[val_idx] += blend_prob / N_REPEATS
        fold_losses.append(log_loss(y_val, blend_prob))

    fold_arr = np.array(fold_losses).reshape(N_REPEATS, N_SPLITS)
    repeat_means = fold_arr.mean(axis=1)
    mean_ll = repeat_means.mean()
    std_ll = repeat_means.std()

    per_match = [-np.log(oof_probs[i, y_enc[i]] + 1e-15) for i in range(n)]
    try:
        _, p_base = wilcoxon(np.array(per_match) - BASELINE_LOSS, alternative='less')
    except Exception:
        p_base = 1.0
    try:
        _, p_front = wilcoxon(np.array(per_match) - FRONTIER_LOSS, alternative='less')
    except Exception:
        p_front = 1.0

    delta_base = mean_ll - BASELINE_LOSS
    verdict = "GREEN" if delta_base < -0.01 and p_base < 0.05 else ("RED" if delta_base > 0.01 else "FLAT")
    acc = accuracy_score(y_enc, np.argmax(oof_probs, axis=1))
    metrics = {
        "experiment": exp_name,
        "cv_log_loss_mean": round(mean_ll, 4),
        "cv_log_loss_std": round(std_ll, 4),
        "accuracy": round(acc, 4),
        "delta_vs_baseline_0.8337": round(delta_base, 4),
        "delta_vs_frontier_0.7608": round(mean_ll - FRONTIER_LOSS, 4),
        "wilcoxon_vs_baseline_pvalue": round(p_base, 4),
        "wilcoxon_vs_frontier_pvalue": round(p_front, 4),
        "verdict_vs_baseline": verdict,
        "label_classes": list(classes),
        "blend_weights": blend_weights,
        "blend_components": [name for name, _ in components],
    }
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"CV log-loss: {mean_ll:.4f} ± {std_ll:.4f}")
        print(f"Δ vs baseline: {delta_base:+.4f} (p={p_base:.4f}) → {verdict}")

    out_dir = os.path.join(ARTIFACTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(out_dir, "oof_probs.npy"), oof_probs)
    pd.DataFrame(oof_probs, columns=classes).to_csv(
        os.path.join(out_dir, "oof_predictions.csv"), index=False)
    return metrics, oof_probs


def run_experiments():
    print("Loading WC-2026 features...")
    df = build_match_features(include_context=False)
    X = df[ELO_FEATURES].values
    y = df["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print(f"n={len(y)}, classes={classes}")

    def make_logreg(C=1.0): return LogisticRegression(C=C, solver='lbfgs', max_iter=500)
    def make_svm(C=1.0, g='scale'): return SVC(C=C, gamma=g, kernel='rbf', probability=True)

    # W3C-11a: TwoStage C-sweep paired with SVM
    print("\n--- W3C-11a: TwoStage C-sweep (with SVM 50/50) ---")
    best_11a = None
    best_11a_C = None
    for C in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        C_label = str(C).replace('.', '')
        m, _ = cv_blend_evaluate(X, y_enc, classes,
            components=[(f"TwoStage(C={C})", lambda C=C: TwoStageModel(C1=C, C2=C)),
                        ("SVM-RBF(C=1.0)", lambda: make_svm(1.0))],
            blend_weights=[0.5, 0.5], exp_name=f"W3C-11a_C{C_label}")
        if best_11a is None or m['cv_log_loss_mean'] < best_11a['cv_log_loss_mean']:
            best_11a = m
            best_11a_C = C
    print(f"Best TwoStage C={best_11a_C}: {best_11a['cv_log_loss_mean']:.4f}")

    # W3C-11b: Optimize weights for LogReg+SVM+TwoStage
    print("\n--- W3C-11b: Weight optimization for LogReg+SVM+TwoStage ---")
    best_11b = None
    best_w = None
    for w_lr in [0.1, 0.2, 0.3, 0.4]:
        for w_svm in [0.3, 0.4, 0.5, 0.6]:
            w_ts = 1.0 - w_lr - w_svm
            if w_ts <= 0:
                continue
            label = f"lr{int(w_lr*10)}_svm{int(w_svm*10)}_ts{int(round(w_ts*10))}"
            m, _ = cv_blend_evaluate(X, y_enc, classes,
                components=[("LogReg(C=1.0)", lambda: make_logreg(1.0)),
                            ("SVM-RBF(C=1.0)", lambda: make_svm(1.0)),
                            ("TwoStage(C=1.0)", lambda: TwoStageModel(C1=1.0, C2=1.0))],
                blend_weights=[w_lr, w_svm, w_ts], exp_name=f"W3C-11b_{label}", verbose=False)
            if best_11b is None or m['cv_log_loss_mean'] < best_11b['cv_log_loss_mean']:
                best_11b = m
                best_w = (w_lr, w_svm, w_ts)
    print(f"Best weights (LogReg={best_w[0]}, SVM={best_w[1]}, TwoStage={best_w[2]:.1f}): {best_11b['cv_log_loss_mean']:.4f}")
    print(f"  Δ={best_11b['delta_vs_baseline_0.8337']:+.4f}, p={best_11b['wilcoxon_vs_baseline_pvalue']:.4f}, {best_11b['verdict_vs_baseline']}")

    # Re-save best_11b under a canonical name
    best_11b['experiment'] = 'W3C-11b_best'
    out_dir = os.path.join(ARTIFACTS_DIR, 'W3C-11b_best')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(best_11b, f, indent=2)

    # W3C-11c: SVM C-sweep (tune SVM component)
    print("\n--- W3C-11c: SVM C-sweep in LogReg+SVM+TwoStage blend ---")
    for C_svm in [0.3, 0.5, 2.0, 3.0]:
        cv_blend_evaluate(X, y_enc, classes,
            components=[("LogReg(C=1.0)", lambda: make_logreg(1.0)),
                        (f"SVM-RBF(C={C_svm})", lambda C=C_svm: make_svm(C)),
                        ("TwoStage(C=1.0)", lambda: TwoStageModel(1.0, 1.0))],
            blend_weights=[1/3, 1/3, 1/3], exp_name=f"W3C-11c_svm{str(C_svm).replace('.','')}")

    print("\nAll W3C-11 experiments done.")


if __name__ == "__main__":
    run_experiments()
