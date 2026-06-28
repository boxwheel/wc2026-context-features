"""
W3C-18: Stage 2 feature engineering + SVM hyperparameter tuning.
  a) SVM C sweep with best KDE (bw=300, pw=0.2)
  b) SVM gamma tuning with best KDE
  c) Stage 2 with additional features: abs(elo_diff), elo_diff^2
  d) Fine prior_weight sweep: {0.12, 0.15, 0.18, 0.20, 0.22, 0.25}
  e) CalibratedClassifierCV instead of SVC(probability=True)
  f) KDE-TS with polynomial Stage 2 features
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
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
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.2, poly_stage2=False):
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight
        self.poly_stage2 = poly_stage2

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        if dm.sum() >= 4:
            X2 = self._stage2_feats(X[dm])
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=1000)
            self.stage2.fit(X2, (y[dm] == 2).astype(int))
        else:
            self.stage2 = None
        return self

    def _stage2_feats(self, X):
        if self.poly_stage2:
            elo_diff = X[:, 0:1]
            abs_elo  = np.abs(elo_diff)
            elo_sq   = elo_diff ** 2
            host     = X[:, 1:2] if X.shape[1] > 1 else np.zeros((len(X), 1))
            return np.hstack([elo_diff, host, abs_elo, elo_sq])
        return X

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
        X2 = self._stage2_feats(X)
        phd = self.stage2.predict_proba(X2)[:, 1] if self.stage2 else np.full(n, 0.5)
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

    # ── W3C-18a: SVM C sweep ────────────────────────────────────────────────
    print("\n--- W3C-18a: SVM C sweep with best KDE ---")
    best_18a = None
    for C in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best),
             ("SVM", lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True))],
            [0.5, 0.5], f"W3C-18a_C{str(C).replace('.','')}", verbose=False)
        print(f"  C={C}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_18a is None or m['cv_log_loss_mean'] < best_18a['cv_log_loss_mean']:
            best_18a = m
    print(f"Best W3C-18a: {best_18a['cv_log_loss_mean']:.4f} → {best_18a['experiment']}")

    # ── W3C-18b: SVM gamma tuning ───────────────────────────────────────────
    print("\n--- W3C-18b: SVM gamma tuning ---")
    best_18b = None
    for g in [0.1, 0.5, 1.0, 2.0, 5.0]:
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best),
             ("SVM", lambda g=g: SVC(C=1.0, kernel='rbf', gamma=g, probability=True))],
            [0.5, 0.5], f"W3C-18b_g{str(g).replace('.','')}", verbose=False)
        print(f"  gamma={g}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_18b is None or m['cv_log_loss_mean'] < best_18b['cv_log_loss_mean']:
            best_18b = m
    print(f"Best W3C-18b: {best_18b['cv_log_loss_mean']:.4f} → {best_18b['experiment']}")

    # ── W3C-18c: Stage 2 polynomial features ───────────────────────────────
    print("\n--- W3C-18c: KDE-TS with poly Stage 2 + SVM ---")
    kde_poly = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2, poly_stage2=True)
    svm_fn   = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
    m18c = cv_blend(X, yenc, classes,
        [("KDE_poly", kde_poly), ("SVM", svm_fn)],
        [0.5, 0.5], "W3C-18c_poly_stage2")

    # ── W3C-18d: Fine prior_weight sweep ───────────────────────────────────
    print("\n--- W3C-18d: Fine prior_weight sweep ---")
    best_18d = None
    for pw in [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30]:
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", lambda pw=pw: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=pw)),
             ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-18d_pw{int(pw*100):02d}", verbose=False)
        print(f"  pw={pw:.2f}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_18d is None or m['cv_log_loss_mean'] < best_18d['cv_log_loss_mean']:
            best_18d = m
    print(f"Best W3C-18d: {best_18d['cv_log_loss_mean']:.4f} → {best_18d['experiment']}")

    # ── W3C-18e: CalibratedClassifierCV SVM ────────────────────────────────
    print("\n--- W3C-18e: CalibratedClassifierCV SVM + KDE-TS ---")
    def calibrated_svm():
        base = SVC(C=1.0, kernel='rbf', gamma='scale')
        return CalibratedClassifierCV(base, cv=5, method='sigmoid')
    m18e = cv_blend(X, yenc, classes,
        [("KDE_TS", kde_best), ("CalibSVM", calibrated_svm)],
        [0.5, 0.5], "W3C-18e_calib_svm")

    # ── W3C-18f: KDE-TS with bw=300, pw=0.2 + best SVM from 18a ───────────
    # Run combined if SVM best C != 1.0
    best_C = float(best_18a['experiment'].split('_C')[1].replace('0', '0.').lstrip('.') if '_C' in best_18a['experiment'] else 1.0)
    # simpler: re-read from experiment name
    exp_c_part = best_18a['experiment'].replace('W3C-18a_C', '')
    if exp_c_part == '10':
        best_C = 10.0
    elif exp_c_part == '01':
        best_C = 0.1
    elif exp_c_part == '03':
        best_C = 0.3
    elif exp_c_part == '05':
        best_C = 0.5
    elif exp_c_part == '10':
        best_C = 1.0
    elif exp_c_part == '20':
        best_C = 2.0
    elif exp_c_part == '50':
        best_C = 5.0
    elif exp_c_part == '100':
        best_C = 10.0
    else:
        best_C = 1.0

    if best_C != 1.0:
        print(f"\n--- W3C-18f: Best C={best_C} from 18a with best KDE ---")
        m18f = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_best),
             ("SVM", lambda: SVC(C=best_C, kernel='rbf', gamma='scale', probability=True))],
            [0.5, 0.5], f"W3C-18f_bestC{str(best_C).replace('.','')}")
    else:
        print(f"\n(Best SVM C=1.0, same as current — skipping W3C-18f)")

    print("\nAll W3C-18 experiments done.")


if __name__ == "__main__":
    run_experiments()
