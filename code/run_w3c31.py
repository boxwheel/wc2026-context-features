"""
W3C-31: Joint alpha x C grid for 5-feat SVM + KDE blend.
W3C-30 found:
  - alpha=0.50, C=0.35 → 0.7781 GREEN (best)
  - alpha=0.70, C=2.0 → 0.7800 GREEN
  - alpha=0.50, C=? unknown
  Joint grid needed to find true optimum.

  a) Joint grid: alpha in {0.30,0.35,0.40,0.45,0.50,0.55,0.60} x C in {0.5,1.0,1.5,2.0,3.0,5.0}
  b) Fine sweep around best joint (alpha, C)
  c) Feature set test at best (alpha, C): is 5-feat still optimal?
  d) Add 6th feature: try adding home_is_host, mv_top11_ratio, or age_diff
  e) SVC with different kernels at best config
"""
import numpy as np
import pandas as pd
import json, os, sys, warnings
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


class KDETwoStageModel:
    """KDE Stage 1 on elo_diff + LogReg Stage 2 on elo_diff + host_advantage."""
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
    y = df["label"].values
    le = LabelEncoder()
    yenc = le.fit_transform(y)
    classes = le.classes_
    print(f"n={len(yenc)}, classes={classes}")

    FEAT5 = ['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff']
    FEAT4 = ['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff']
    FEAT3MV = ['elo_diff', 'host_advantage', 'mv_top11_diff']
    X5 = df[FEAT5].values
    X4 = df[FEAT4].values
    X3mv = df[FEAT3MV].values

    kde_fn = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)

    # ── W3C-31a: Joint alpha x C grid on 5-feat ──────────────────────────────
    print("\n--- W3C-31a: Joint alpha x C grid (5-feat SVM) ---")
    best_31a = None
    all_31a = []
    for alpha in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        for C in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            svm_c = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X5, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM5", svm_c)],
                [alpha, 1 - alpha], f"W3C-31a_a{int(alpha*100)}_C{int(C*10):02d}", verbose=False)
            all_31a.append(m)
            if best_31a is None or m['cv_log_loss_mean'] < best_31a['cv_log_loss_mean']:
                best_31a = m
        # Print row summary for this alpha
        row = [x for x in all_31a if f"_a{int(alpha*100)}_" in x['experiment']]
        best_row = min(row, key=lambda x: x['cv_log_loss_mean'])
        print(f"  alpha={alpha}: best C → {best_row['cv_log_loss_mean']:.4f} at {best_row['experiment']}")
    print(f"Best W3C-31a: {best_31a['cv_log_loss_mean']:.4f} → {best_31a['experiment']}")

    # Parse best alpha and C from best_31a
    import re
    m_best = re.search(r'a(\d+)_C(\d+)', best_31a['experiment'])
    best_alpha = int(m_best.group(1)) / 100
    best_C = int(m_best.group(2)) / 10
    print(f"Best params: alpha={best_alpha}, C={best_C}")

    # ── W3C-31b: Fine-grain around best (alpha, C) ────────────────────────────
    print(f"\n--- W3C-31b: Fine sweep around best alpha={best_alpha}, C={best_C} ---")
    best_31b = None
    alpha_range = sorted(set([max(0.1, best_alpha - 0.1), best_alpha - 0.05, best_alpha,
                              best_alpha + 0.05, min(0.9, best_alpha + 0.1)]))
    C_range = sorted(set([max(0.1, best_C * 0.5), best_C * 0.7, best_C, best_C * 1.3, best_C * 2.0]))
    for alpha in alpha_range:
        for C in C_range:
            svm_c = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            tag = f"a{int(alpha*100)}_C{int(C*10):02d}"
            m = cv_blend(X5, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM5", svm_c)],
                [alpha, 1 - alpha], f"W3C-31b_{tag}", verbose=False)
            print(f"  alpha={alpha:.2f}, C={C:.1f}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
            if best_31b is None or m['cv_log_loss_mean'] < best_31b['cv_log_loss_mean']:
                best_31b = m
    print(f"Best W3C-31b: {best_31b['cv_log_loss_mean']:.4f} → {best_31b['experiment']}")

    # ── W3C-31c: Feature set test at best (alpha, C) ─────────────────────────
    print(f"\n--- W3C-31c: Feature subset test at best config ---")
    svm_best = lambda: SVC(C=best_C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    best_31c = None
    for feats, tag in [
        (FEAT3MV, "3feat_mv"),
        (FEAT4, "4feat"),
        (FEAT5, "5feat"),
    ]:
        X_f = df[feats].values
        m = cv_blend(X_f, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM", svm_best)],
            [best_alpha, 1 - best_alpha], f"W3C-31c_{tag}", verbose=False)
        print(f"  {tag}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_31c is None or m['cv_log_loss_mean'] < best_31c['cv_log_loss_mean']:
            best_31c = m
    print(f"Best W3C-31c: {best_31c['cv_log_loss_mean']:.4f}")

    # ── W3C-31d: Try 6th feature at best config ───────────────────────────────
    print(f"\n--- W3C-31d: 6th feature exploration at best config ---")
    best_31d = None
    for feat6, tag in [
        ('home_is_host', "host6"),
        ('stage_enc', "stage6"),
        ('home_fifa_ranking_pre_tournament', "hrank6"),
        ('away_fifa_ranking_pre_tournament', "arank6"),
    ]:
        try:
            X6 = df[FEAT5 + [feat6]].values
            svm6 = lambda C=best_C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X6, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM6", svm6)],
                [best_alpha, 1 - best_alpha], f"W3C-31d_{tag}", verbose=False)
            print(f"  +{feat6}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
            if best_31d is None or m['cv_log_loss_mean'] < best_31d['cv_log_loss_mean']:
                best_31d = m
        except Exception as e:
            print(f"  +{feat6}: ERROR {e}")
    if best_31d:
        print(f"Best W3C-31d: {best_31d['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-31 experiments done.")
    all_bests = [x for x in [best_31a, best_31b, best_31c, best_31d] if x is not None]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-31 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")

    import json
    results = sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])
    with open("/tmp/w3c31_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "note": "Joint alpha-C grid for 5-feat SVM",
                   "all_bests": results}, f, indent=2)
    print("Summary saved to /tmp/w3c31_summary.json")


if __name__ == "__main__":
    run_experiments()
