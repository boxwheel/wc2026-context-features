"""
W3C-33: SVM kernel variants + calibration + three-model ensembles.
W3C-32 showed: only RBF-SVM works; RF/GBM/MLR/KNN all much worse.
Current frontier: 0.7744 (alpha=0.44, C=1.7) / 0.7745 (alpha=0.45, C=2.0) GREEN.

Hypotheses:
  a) Polynomial SVM (degree=2,3) as complement to KDE-TS
  b) Fixed gamma values for RBF SVM (not 'scale') — find best gamma
  c) CalibratedClassifierCV(SVC, isotonic) for better probability calibration
  d) ExtraTreesClassifier (random splits, different from RF, might generalize better)
  e) Three-way blend: KDE + SVM(C=1.7) + SVM(C=3.0) to diversify SVM predictions
  f) SVM with feature interactions via PolynomialFeatures(degree=2) pre-processing
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
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
RECORD_LOSS = 0.7744  # W3C-32a best


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


class PolyFeatSVM:
    """SVM on polynomial-expanded features (degree=2)."""
    def __init__(self, C=2.0, degree=2, random_state=0):
        self.C = C; self.degree = degree; self.random_state = random_state
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.svm = SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=random_state)

    def fit(self, X, y):
        Xp = self.poly.fit_transform(X)
        self.svm.fit(Xp, y)
        return self

    def predict_proba(self, X):
        return self.svm.predict_proba(self.poly.transform(X))


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
           "delta_vs_record_0.7744": round(ml - RECORD_LOSS, 4),
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
    svm_best = lambda: SVC(C=2.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED)

    # ── W3C-33a: Polynomial SVM (degree=2,3) as complement ───────────────────
    print("\n--- W3C-33a: Polynomial SVM complement at alpha=0.45 ---")
    best_33a = None
    for degree in [2, 3]:
        for C in [0.5, 1.0, 2.0, 5.0]:
            poly_fn = lambda C=C, d=degree: PolyFeatSVM(C=C, degree=d, random_state=SEED)
            tag = f"deg{degree}_C{int(C*10):02d}"
            m = cv_blend(X5, yenc, classes,
                [("KDE_TS", kde_fn), ("PolySVM", poly_fn)],
                [0.45, 0.55], f"W3C-33a_{tag}", verbose=False)
            print(f"  PolySVM d={degree} C={C}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
            if best_33a is None or m['cv_log_loss_mean'] < best_33a['cv_log_loss_mean']:
                best_33a = m
    print(f"Best W3C-33a: {best_33a['cv_log_loss_mean']:.4f}")

    # ── W3C-33b: RBF SVM with fixed gamma values ─────────────────────────────
    print("\n--- W3C-33b: RBF SVM fixed gamma at alpha=0.45, C=2.0 ---")
    best_33b = None
    for gamma in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 'scale', 'auto']:
        tag = f"g{str(gamma).replace('.','p')}"
        svm_g = lambda g=gamma: SVC(C=2.0, kernel='rbf', gamma=g, probability=True, random_state=SEED)
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM_g", svm_g)],
            [0.45, 0.55], f"W3C-33b_{tag}", verbose=False)
        print(f"  gamma={gamma}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
        if best_33b is None or m['cv_log_loss_mean'] < best_33b['cv_log_loss_mean']:
            best_33b = m
    # Also try gamma sweep at alpha=0.44, C=1.7 (previously best)
    for gamma in [0.1, 0.2, 'scale']:
        tag = f"a44_g{str(gamma).replace('.','p')}"
        svm_g = lambda g=gamma: SVC(C=1.7, kernel='rbf', gamma=g, probability=True, random_state=SEED)
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM_g", svm_g)],
            [0.44, 0.56], f"W3C-33b_{tag}", verbose=False)
        print(f"  alpha=0.44 C=1.7 gamma={gamma}: {m['cv_log_loss_mean']:.4f} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
        if m['cv_log_loss_mean'] < best_33b['cv_log_loss_mean']:
            best_33b = m
    print(f"Best W3C-33b: {best_33b['cv_log_loss_mean']:.4f}")

    # ── W3C-33c: Isotonic-calibrated SVM (CalibratedClassifierCV) ───────────
    print("\n--- W3C-33c: Isotonic-calibrated SVM at alpha=0.45 ---")
    best_33c = None
    for C, cv_cal in [(2.0, 3), (2.0, 5), (1.7, 3), (1.7, 5)]:
        cal_fn = lambda C=C, cv=cv_cal: CalibratedClassifierCV(
            SVC(C=C, kernel='rbf', gamma='scale', random_state=SEED),
            method='isotonic', cv=cv)
        tag = f"C{int(C*10):02d}_cv{cv_cal}"
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("CalSVM", cal_fn)],
            [0.45, 0.55], f"W3C-33c_{tag}", verbose=False)
        print(f"  Isotonic C={C} cv={cv_cal}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
        if best_33c is None or m['cv_log_loss_mean'] < best_33c['cv_log_loss_mean']:
            best_33c = m
    # Also sigmoid calibration
    for C in [1.7, 2.0]:
        cal_fn = lambda C=C: CalibratedClassifierCV(
            SVC(C=C, kernel='rbf', gamma='scale', random_state=SEED),
            method='sigmoid', cv=3)
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("SigSVM", cal_fn)],
            [0.45, 0.55], f"W3C-33c_sig_C{int(C*10):02d}", verbose=False)
        print(f"  Sigmoid C={C}: {m['cv_log_loss_mean']:.4f} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
        if m['cv_log_loss_mean'] < best_33c['cv_log_loss_mean']:
            best_33c = m
    print(f"Best W3C-33c: {best_33c['cv_log_loss_mean']:.4f}")

    # ── W3C-33d: ExtraTreesClassifier complement ──────────────────────────────
    print("\n--- W3C-33d: ExtraTrees complement at alpha=0.45 ---")
    best_33d = None
    for depth, ne in [(3, 200), (3, 500), (4, 200), (None, 200)]:
        et_fn = lambda d=depth, n=ne: ExtraTreesClassifier(
            max_depth=d, n_estimators=n, random_state=SEED, n_jobs=-1)
        tag = f"d{depth}n{ne}"
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("ET5", et_fn)],
            [0.45, 0.55], f"W3C-33d_{tag}", verbose=False)
        print(f"  ET d={depth} n={ne}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_33d is None or m['cv_log_loss_mean'] < best_33d['cv_log_loss_mean']:
            best_33d = m
    print(f"Best W3C-33d: {best_33d['cv_log_loss_mean']:.4f}")

    # ── W3C-33e: Three-way blend: KDE + SVM(C=1.7) + SVM(C=3.0) ─────────────
    print("\n--- W3C-33e: Three-way SVM ensemble ---")
    best_33e = None
    svm_a = lambda: SVC(C=1.7, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    svm_b = lambda: SVC(C=3.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    svm_c_fn = lambda: SVC(C=2.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    for w_kde, w_a, w_b in [
        (0.40, 0.35, 0.25),
        (0.40, 0.40, 0.20),
        (0.45, 0.30, 0.25),
        (0.45, 0.35, 0.20),
        (0.45, 0.275, 0.275),
        (0.50, 0.25, 0.25),
    ]:
        tag = f"k{int(w_kde*100)}_a{int(w_a*100)}_b{int(w_b*100)}"
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM_a", svm_a), ("SVM_b", svm_b)],
            [w_kde, w_a, w_b], f"W3C-33e_{tag}", verbose=False)
        print(f"  KDE={w_kde} SVM(C=1.7)={w_a} SVM(C=3.0)={w_b}: {m['cv_log_loss_mean']:.4f} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
        if best_33e is None or m['cv_log_loss_mean'] < best_33e['cv_log_loss_mean']:
            best_33e = m
    # Try with C=2.0 and C=1.4 pair
    svm_d = lambda: SVC(C=1.4, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    for w_kde, w_a, w_b in [(0.45, 0.30, 0.25), (0.45, 0.275, 0.275)]:
        tag = f"C14_k{int(w_kde*100)}_a{int(w_a*100)}_b{int(w_b*100)}"
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM14", svm_d), ("SVM20", svm_c_fn)],
            [w_kde, w_a, w_b], f"W3C-33e_{tag}", verbose=False)
        print(f"  KDE={w_kde} SVM(C=1.4)={w_a} SVM(C=2.0)={w_b}: {m['cv_log_loss_mean']:.4f} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
        if m['cv_log_loss_mean'] < best_33e['cv_log_loss_mean']:
            best_33e = m
    print(f"Best W3C-33e: {best_33e['cv_log_loss_mean']:.4f}")

    # ── W3C-33f: SVM on interaction-augmented features (hand-crafted) ────────
    print("\n--- W3C-33f: SVM on hand-crafted feature interactions ---")
    # Add: elo_diff*rank_diff, elo_diff*mv_top11_diff, rank_diff*mv_top11_diff
    # to stay interpretable and avoid explosion
    elo = df['elo_diff'].values
    rank = df['rank_diff'].values
    mv = df['mv_top11_diff'].values
    gk = df['gk_mv_diff'].values
    host = df['host_advantage'].values
    # Interaction features
    X_int = np.column_stack([
        elo, host, rank, mv, gk,
        elo * rank / (np.std(elo) * np.std(rank) + 1e-9),
        elo * mv / (np.std(elo) * np.std(mv) + 1e-9),
        rank * mv / (np.std(rank) * np.std(mv) + 1e-9),
    ])
    best_33f = None
    for C in [0.5, 1.0, 2.0]:
        for alpha in [0.40, 0.45, 0.50]:
            svm_int = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X_int, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM_int", svm_int)],
                [alpha, 1 - alpha], f"W3C-33f_C{int(C*10):02d}_a{int(alpha*100)}", verbose=False)
            print(f"  Interact C={C} a={alpha}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
            if best_33f is None or m['cv_log_loss_mean'] < best_33f['cv_log_loss_mean']:
                best_33f = m
    print(f"Best W3C-33f: {best_33f['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-33 experiments done.")
    all_bests = [best_33a, best_33b, best_33c, best_33d, best_33e, best_33f]
    all_bests = [x for x in all_bests if x is not None]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-33 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")
    print(f"Previous record: {RECORD_LOSS:.4f}, delta: {overall['cv_log_loss_mean'] - RECORD_LOSS:+.4f}")

    with open("/tmp/w3c33_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "previous_record": RECORD_LOSS,
                   "note": "SVM kernel variants, calibration, ensemble, interactions",
                   "all_bests": sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])}, f, indent=2)
    print("Summary saved to /tmp/w3c33_summary.json")


if __name__ == "__main__":
    run_experiments()
