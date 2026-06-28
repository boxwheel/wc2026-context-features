"""
W3C-27: Follow-up on W3C-26 finding that alpha=0.70 → 0.7970 (new record).
The KDE-weight trend was still decreasing at alpha=0.70 with SVC(random_state=0).
  a) Push alpha to {0.75, 0.80, 0.85, 0.90, 0.95, 1.0} — Gaussian + SVM(rs=0)
  b) Push alpha to {0.75, 0.80, 0.85, 0.90, 0.95, 1.0} — Student-t(nu=10,bw=200) + SVM(rs=0)
  c) SVM-C sweep at alpha=0.70: C in {0.3, 0.5, 1.0, 2.0, 5.0} for Student-t blend
  d) nu fine-tune at alpha=0.70: nu in {5, 8, 10, 12, 15, 20} Student-t + SVM(rs=0)
  e) 3-way blend: Student-t(a_st) + Gaussian(a_g) + SVM(1-a_st-a_g)
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
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


class GaussianKDETwoStageModel:
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.2):
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
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
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
        p = np.stack([pdec * (1 - phd), p_draw, pdec * phd], axis=1)
        return p / p.sum(axis=1, keepdims=True)


class StudentTKDETwoStageModel:
    def __init__(self, bw=200.0, nu=10, C2=2.0, prior_weight=0.2):
        self.bw = bw
        self.nu = nu
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
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
        elo, n = X[:, 0], len(X)
        p_draw = np.zeros(n)
        for j in range(n):
            t2 = ((elo[j] - self.X_train) / self.bw) ** 2
            w = (1.0 + t2 / self.nu) ** (-(self.nu + 1) / 2.0)
            ws = w.sum()
            if ws > 0:
                kde = np.dot(w, self.y_draw) / ws
                p_draw[j] = (1 - self.prior_weight) * np.clip(kde, 0.05, 0.95) + self.prior_weight * self.prior_draw
            else:
                p_draw[j] = self.prior_draw
        pdec = 1.0 - p_draw
        phd = self.stage2.predict_proba(X)[:, 1] if self.stage2 else np.full(n, 0.5)
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

    svm_fn = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED)

    # ── W3C-27a: Push Gaussian alpha to 1.0 ─────────────────────────────────
    print("\n--- W3C-27a: Gaussian KDE alpha push to 1.0 ---")
    best_27a = None
    for alpha in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        gauss_fn = lambda: GaussianKDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)
        comps = [("GaussKDE", gauss_fn), ("SVM", svm_fn)] if alpha < 1.0 else [("GaussKDE", gauss_fn)]
        wts = [alpha, 1 - alpha] if alpha < 1.0 else [1.0]
        m = cv_blend(X, yenc, classes, comps, wts, f"W3C-27a_gauss_a{int(alpha*100)}", verbose=False)
        print(f"  alpha(KDE)={alpha:.2f}: {m['cv_log_loss_mean']:.4f} ± {m['cv_log_loss_std']:.4f} → {m['verdict_vs_baseline']}")
        if best_27a is None or m['cv_log_loss_mean'] < best_27a['cv_log_loss_mean']:
            best_27a = m
    print(f"Best W3C-27a: {best_27a['cv_log_loss_mean']:.4f} → {best_27a['experiment']}")

    # ── W3C-27b: Push Student-t alpha to 1.0 ────────────────────────────────
    print("\n--- W3C-27b: Student-t(nu=10,bw=200) alpha push to 1.0 ---")
    best_27b = None
    for alpha in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        st_fn = lambda: StudentTKDETwoStageModel(bw=200.0, nu=10, C2=2.0, prior_weight=0.2)
        comps = [("StKDE", st_fn), ("SVM", svm_fn)] if alpha < 1.0 else [("StKDE", st_fn)]
        wts = [alpha, 1 - alpha] if alpha < 1.0 else [1.0]
        m = cv_blend(X, yenc, classes, comps, wts, f"W3C-27b_st_a{int(alpha*100)}", verbose=False)
        print(f"  alpha(KDE)={alpha:.2f}: {m['cv_log_loss_mean']:.4f} ± {m['cv_log_loss_std']:.4f} → {m['verdict_vs_baseline']}")
        if best_27b is None or m['cv_log_loss_mean'] < best_27b['cv_log_loss_mean']:
            best_27b = m
    print(f"Best W3C-27b: {best_27b['cv_log_loss_mean']:.4f} → {best_27b['experiment']}")

    # ── W3C-27c: SVM-C sweep at alpha=0.70 for Student-t ────────────────────
    print("\n--- W3C-27c: SVM-C sweep at alpha=0.70, Student-t(nu=10,bw=200) ---")
    st_fn = lambda: StudentTKDETwoStageModel(bw=200.0, nu=10, C2=2.0, prior_weight=0.2)
    best_27c = None
    for C in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        svm_c = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn), ("SVM", svm_c)],
            [0.70, 0.30], f"W3C-27c_C{int(C*10):02d}", verbose=False)
        print(f"  C={C}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_27c is None or m['cv_log_loss_mean'] < best_27c['cv_log_loss_mean']:
            best_27c = m
    print(f"Best W3C-27c: {best_27c['cv_log_loss_mean']:.4f} → {best_27c['experiment']}")

    # ── W3C-27d: nu sweep at alpha=0.70, Student-t + SVM(rs=0) ──────────────
    print("\n--- W3C-27d: nu sweep at alpha=0.70, Student-t(bw=200)+SVM ---")
    best_27d = None
    for nu in [3, 5, 8, 10, 12, 15, 20, 30]:
        st_fn2 = lambda nu=nu: StudentTKDETwoStageModel(bw=200.0, nu=nu, C2=2.0, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn2), ("SVM", svm_fn)],
            [0.70, 0.30], f"W3C-27d_nu{nu}_a70", verbose=False)
        print(f"  nu={nu}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_27d is None or m['cv_log_loss_mean'] < best_27d['cv_log_loss_mean']:
            best_27d = m
    print(f"Best W3C-27d: {best_27d['cv_log_loss_mean']:.4f} → {best_27d['experiment']}")

    # ── W3C-27e: 3-way blend: Gauss + Student-t + SVM ──────────────────────
    print("\n--- W3C-27e: 3-way blend (Gauss + St-t + SVM) ---")
    gauss_fn = lambda: GaussianKDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)
    st_fn = lambda: StudentTKDETwoStageModel(bw=200.0, nu=10, C2=2.0, prior_weight=0.2)
    best_27e = None
    for ag, ast, asvm in [
        (0.35, 0.35, 0.30),
        (0.40, 0.30, 0.30),
        (0.30, 0.40, 0.30),
        (0.35, 0.45, 0.20),
        (0.45, 0.35, 0.20),
        (0.40, 0.40, 0.20),
        (0.50, 0.30, 0.20),
        (0.30, 0.50, 0.20),
    ]:
        tag = f"g{int(ag*100)}s{int(ast*100)}v{int(asvm*100)}"
        m = cv_blend(X, yenc, classes,
            [("Gauss", gauss_fn), ("StKDE", st_fn), ("SVM", svm_fn)],
            [ag, ast, asvm], f"W3C-27e_{tag}", verbose=False)
        print(f"  Gauss={ag} St-t={ast} SVM={asvm}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_27e is None or m['cv_log_loss_mean'] < best_27e['cv_log_loss_mean']:
            best_27e = m
    print(f"Best W3C-27e: {best_27e['cv_log_loss_mean']:.4f} → {best_27e['experiment']}")

    print("\nAll W3C-27 experiments done.")
    all_bests = [best_27a, best_27b, best_27c, best_27d, best_27e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-27 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")

    # Save summary
    results = []
    for b in all_bests:
        results.append(b)
    results.sort(key=lambda x: x['cv_log_loss_mean'])
    import json
    with open("/tmp/w3c27_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "note": "W3C-26 showed 0.7970 with alpha=0.70 and SVC(rs=0). Pushing alpha further.",
                   "all_bests": results}, f, indent=2)
    print("Summary saved to /tmp/w3c27_summary.json")


if __name__ == "__main__":
    run_experiments()
