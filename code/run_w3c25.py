"""
W3C-25: Student-t kernel fine-tuning after W3C-24c found nu=10, bw=200 = 0.7988 (new best).
  a) Fine nu sweep: {8, 10, 12, 15, 20, 50, 100} at bw=200
  b) Fine bw sweep: {125, 150, 175, 200, 225, 250, 275} at nu=10
  c) Joint (nu, bw) grid around best
  d) prior_weight sweep for best Student-t config
  e) Student-t(best) + SVM blend weight sweep (alpha)
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


class StudentTKDETwoStageModel:
    """KDE Stage 1 with Student-t kernel."""
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

    svm_fn = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)

    # ── W3C-25a: Fine nu sweep at bw=200 ────────────────────────────────────
    print("\n--- W3C-25a: Fine nu sweep at bw=200 ---")
    best_25a = None
    for nu in [5, 8, 10, 12, 15, 20, 30, 50]:
        st_fn = lambda nu=nu: StudentTKDETwoStageModel(bw=200, nu=nu, C2=2.0, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn), ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-25a_nu{nu}_bw200", verbose=False)
        print(f"  nu={nu}: {m['cv_log_loss_mean']:.4f} ± {m['cv_log_loss_std']:.4f} → {m['verdict_vs_baseline']}")
        if best_25a is None or m['cv_log_loss_mean'] < best_25a['cv_log_loss_mean']:
            best_25a = m
    print(f"Best W3C-25a: {best_25a['cv_log_loss_mean']:.4f} → {best_25a['experiment']}")

    # ── W3C-25b: Fine bw sweep at nu=10 ─────────────────────────────────────
    print("\n--- W3C-25b: Fine bw sweep at nu=10 ---")
    best_25b = None
    for bw in [100, 125, 150, 175, 200, 225, 250, 275, 300]:
        st_fn = lambda bw=bw: StudentTKDETwoStageModel(bw=bw, nu=10, C2=2.0, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn), ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-25b_nu10_bw{bw}", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} ± {m['cv_log_loss_std']:.4f} → {m['verdict_vs_baseline']}")
        if best_25b is None or m['cv_log_loss_mean'] < best_25b['cv_log_loss_mean']:
            best_25b = m
    print(f"Best W3C-25b: {best_25b['cv_log_loss_mean']:.4f} → {best_25b['experiment']}")

    # ── W3C-25c: Joint (nu, bw) grid ────────────────────────────────────────
    print("\n--- W3C-25c: Joint (nu, bw) fine grid ---")
    best_25c = None
    for nu in [8, 10, 12, 15]:
        for bw in [150, 175, 200, 225, 250]:
            st_fn = lambda nu=nu, bw=bw: StudentTKDETwoStageModel(bw=bw, nu=nu, C2=2.0, prior_weight=0.2)
            m = cv_blend(X, yenc, classes,
                [("StKDE", st_fn), ("SVM", svm_fn)],
                [0.5, 0.5], f"W3C-25c_nu{nu}_bw{bw}", verbose=False)
            print(f"  (nu={nu}, bw={bw}): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
            if best_25c is None or m['cv_log_loss_mean'] < best_25c['cv_log_loss_mean']:
                best_25c = m
    print(f"Best W3C-25c: {best_25c['cv_log_loss_mean']:.4f} → {best_25c['experiment']}")

    # ── W3C-25d: prior_weight sweep for Student-t(nu=10, bw=200) ─────────────
    print("\n--- W3C-25d: prior_weight sweep for Student-t(nu=10, bw=200) ---")
    best_25d = None
    for pw in [0.10, 0.15, 0.20, 0.25, 0.30]:
        st_fn = lambda pw=pw: StudentTKDETwoStageModel(bw=200, nu=10, C2=2.0, prior_weight=pw)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn), ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-25d_pw{int(pw*100):02d}", verbose=False)
        print(f"  pw={pw:.2f}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_25d is None or m['cv_log_loss_mean'] < best_25d['cv_log_loss_mean']:
            best_25d = m
    print(f"Best W3C-25d: {best_25d['cv_log_loss_mean']:.4f}")

    # ── W3C-25e: Blend weight (alpha) sweep for best Student-t ───────────────
    print("\n--- W3C-25e: Alpha sweep for Student-t(nu=10, bw=200) ---")
    st_best = lambda: StudentTKDETwoStageModel(bw=200, nu=10, C2=2.0, prior_weight=0.2)
    best_25e = None
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_best), ("SVM", svm_fn)],
            [alpha, 1 - alpha], f"W3C-25e_alpha{int(alpha*10)}", verbose=False)
        print(f"  alpha(StKDE)={alpha}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_25e is None or m['cv_log_loss_mean'] < best_25e['cv_log_loss_mean']:
            best_25e = m
    print(f"Best W3C-25e: {best_25e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-25 experiments done.")
    all_bests = [best_25a, best_25b, best_25c, best_25d, best_25e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-25 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")


if __name__ == "__main__":
    run_experiments()
