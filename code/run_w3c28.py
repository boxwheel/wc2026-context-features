"""
W3C-28: Follow-up on W3C-27 findings.
  - W3C-27c: C=0.3 gave 0.7962 (FLAT) — fine-tune C around 0.3
  - W3C-27a/b: alpha=0.70 is optimal plateau — explore around it
  a) Fine C sweep: {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50} at alpha=0.70, Student-t
  b) SVM gamma sweep: gamma in {0.05, 0.1, 0.2, 'auto', 'scale'} at alpha=0.70, C=0.3
  c) LogReg as complement: replace SVM with LR(C sweep) at alpha=0.70
  d) Gaussian bw sweep at alpha=0.70 with SVM(C=0.3, rs=0)
  e) Prior_weight sweep at alpha=0.70, SVM(C=0.3, rs=0)
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
        self.bw = bw; self.C2 = C2; self.prior_weight = prior_weight

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
        self.bw = bw; self.nu = nu; self.C2 = C2; self.prior_weight = prior_weight

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

    st_fn = lambda: StudentTKDETwoStageModel(bw=200.0, nu=10, C2=2.0, prior_weight=0.2)
    gauss_fn = lambda: GaussianKDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)

    # ── W3C-28a: Fine C sweep around C=0.3, Student-t + SVM, alpha=0.70 ──────
    print("\n--- W3C-28a: Fine C sweep (0.20-0.50), Student-t alpha=0.70 ---")
    best_28a = None
    for C in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]:
        tag = f"C{int(C*100):03d}"
        svm_c = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn), ("SVM", svm_c)],
            [0.70, 0.30], f"W3C-28a_{tag}", verbose=False)
        print(f"  C={C:.2f}: {m['cv_log_loss_mean']:.4f} ± {m['cv_log_loss_std']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        if best_28a is None or m['cv_log_loss_mean'] < best_28a['cv_log_loss_mean']:
            best_28a = m
    print(f"Best W3C-28a: {best_28a['cv_log_loss_mean']:.4f} → {best_28a['experiment']}")

    # ── W3C-28b: Gamma sweep at alpha=0.70, C=0.3 ────────────────────────────
    print("\n--- W3C-28b: SVM gamma sweep at alpha=0.70, C=0.3 ---")
    best_28b = None
    for gamma in ['scale', 'auto', 0.05, 0.10, 0.20, 0.50]:
        tag = f"g{'sc' if gamma=='scale' else ('au' if gamma=='auto' else int(gamma*100))}"
        svm_g = lambda gamma=gamma: SVC(C=0.3, kernel='rbf', gamma=gamma, probability=True, random_state=SEED)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn), ("SVM", svm_g)],
            [0.70, 0.30], f"W3C-28b_{tag}", verbose=False)
        print(f"  gamma={gamma}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        if best_28b is None or m['cv_log_loss_mean'] < best_28b['cv_log_loss_mean']:
            best_28b = m
    print(f"Best W3C-28b: {best_28b['cv_log_loss_mean']:.4f}")

    # ── W3C-28c: LogReg as complement at alpha=0.70 ───────────────────────────
    print("\n--- W3C-28c: LogisticRegression complement at alpha=0.70 ---")
    best_28c = None
    for C in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        lr_fn2 = lambda C=C: LogisticRegression(C=C, solver='lbfgs', max_iter=1000, random_state=SEED)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn), ("LR", lr_fn2)],
            [0.70, 0.30], f"W3C-28c_LR_C{int(C*10):02d}", verbose=False)
        print(f"  LR C={C}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        if best_28c is None or m['cv_log_loss_mean'] < best_28c['cv_log_loss_mean']:
            best_28c = m
    print(f"Best W3C-28c: {best_28c['cv_log_loss_mean']:.4f}")

    # ── W3C-28d: Gaussian bw sweep at alpha=0.70, SVM(C=0.3, rs=0) ───────────
    print("\n--- W3C-28d: Gaussian bw sweep at alpha=0.70, C=0.3 ---")
    svm_best = lambda: SVC(C=0.3, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    best_28d = None
    for bw in [150, 200, 250, 300, 350, 400, 500]:
        gauss_bw = lambda bw=bw: GaussianKDETwoStageModel(bw=bw, C2=2.0, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("GaussKDE", gauss_bw), ("SVM", svm_best)],
            [0.70, 0.30], f"W3C-28d_bw{bw}_C03", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_28d is None or m['cv_log_loss_mean'] < best_28d['cv_log_loss_mean']:
            best_28d = m
    print(f"Best W3C-28d: {best_28d['cv_log_loss_mean']:.4f}")

    # ── W3C-28e: Prior_weight sweep at alpha=0.70, C=0.3 ─────────────────────
    print("\n--- W3C-28e: Prior_weight sweep at alpha=0.70, C=0.3 ---")
    svm_best = lambda: SVC(C=0.3, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    best_28e = None
    for pw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        st_pw = lambda pw=pw: StudentTKDETwoStageModel(bw=200.0, nu=10, C2=2.0, prior_weight=pw)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_pw), ("SVM", svm_best)],
            [0.70, 0.30], f"W3C-28e_pw{int(pw*100):02d}", verbose=False)
        print(f"  pw={pw:.2f}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_28e is None or m['cv_log_loss_mean'] < best_28e['cv_log_loss_mean']:
            best_28e = m
    print(f"Best W3C-28e: {best_28e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-28 experiments done.")
    all_bests = [best_28a, best_28b, best_28c, best_28d, best_28e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-28 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")

    import json
    results = sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])
    with open("/tmp/w3c28_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "note": "Fine-tuning around C=0.3, alpha=0.70 configuration.",
                   "all_bests": results}, f, indent=2)
    print("Summary saved to /tmp/w3c28_summary.json")


if __name__ == "__main__":
    run_experiments()
