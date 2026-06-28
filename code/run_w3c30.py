"""
W3C-30: BREAKTHROUGH — 5-feature SVM gives 0.7822 GREEN (vs 0.7970 with 2 features).
SVM was feature-starved with only elo_diff + host_advantage.
Now exploit: rank_diff, mv_top11_diff, gk_mv_diff added to SVM input.

  a) Alpha sweep: KDE(2-feat, alpha=0.70) + SVM(5-feat), alpha in [0.5..0.9]
  b) SVM-C sweep on 5-feat SVM at best alpha
  c) SVM alone (alpha=0) on 5-feat vs best blend
  d) Feature subset for SVM: try different combos of the 5 features
  e) KDE on elo_diff, alpha=0.70 with SVM on all available features (more than 5)
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


class KDETwoStageModel:
    """KDE Stage 1 (always uses col 0 = elo_diff) + LogReg Stage 2."""
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.2):
        self.bw = bw; self.C2 = C2; self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        dm = (y != 1)
        if dm.sum() >= 4:
            self.stage2 = LogisticRegression(C=self.C2, solver='lbfgs', max_iter=1000)
            self.stage2.fit(X[dm, :2], (y[dm] == 2).astype(int))  # Stage2 uses cols 0,1
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

    # Feature sets
    FEAT5 = ['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff']
    FEAT7 = ['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff', 'caps_diff', 'age_diff']
    FEAT_MORE = ['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff',
                 'home_squad_mean_age', 'away_squad_mean_age', 'home_squad_mean_caps', 'away_squad_mean_caps']

    X5 = df[FEAT5].values
    X7 = df[FEAT7].values
    X2 = df[ELO_FEATURES].values  # baseline 2-feat for KDE
    print(f"5-feat matrix shape: {X5.shape}")

    kde_fn = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)

    # ── W3C-30a: Alpha sweep with 5-feat SVM + 2-feat KDE ────────────────────
    print("\n--- W3C-30a: Alpha sweep: KDE(2-feat) + SVM(5-feat, C=0.35) ---")
    best_30a = None
    for alpha in [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]:
        svm5 = lambda C=0.35: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
        comps = [("KDE_TS", kde_fn), ("SVM5", svm5)] if alpha > 0 else [("SVM5", svm5)]
        wts = [alpha, 1 - alpha] if alpha > 0 else [1.0]
        m = cv_blend(X5, yenc, classes, comps, wts, f"W3C-30a_alpha{int(alpha*100)}", verbose=False)
        print(f"  alpha(KDE)={alpha:.2f}: {m['cv_log_loss_mean']:.4f} ± {m['cv_log_loss_std']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        if best_30a is None or m['cv_log_loss_mean'] < best_30a['cv_log_loss_mean']:
            best_30a = m
    print(f"Best W3C-30a: {best_30a['cv_log_loss_mean']:.4f} → {best_30a['experiment']}")

    # ── W3C-30b: SVM-C sweep at best alpha with 5-feat SVM ───────────────────
    print("\n--- W3C-30b: SVM-C sweep with 5-feat SVM + KDE ---")
    best_30b = None
    for C in [0.1, 0.2, 0.3, 0.35, 0.5, 1.0, 2.0]:
        svm_c = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
        m = cv_blend(X5, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM5", svm_c)],
            [0.70, 0.30], f"W3C-30b_C{int(C*10):02d}", verbose=False)
        print(f"  C={C}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        if best_30b is None or m['cv_log_loss_mean'] < best_30b['cv_log_loss_mean']:
            best_30b = m
    print(f"Best W3C-30b: {best_30b['cv_log_loss_mean']:.4f} → {best_30b['experiment']}")

    # ── W3C-30c: SVM alone on 5 features (no KDE) ────────────────────────────
    print("\n--- W3C-30c: SVM alone on 5 features (no KDE blend) ---")
    best_30c = None
    for C in [0.1, 0.2, 0.3, 0.35, 0.5, 1.0, 2.0, 5.0]:
        svm_only = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
        m = cv_blend(X5, yenc, classes,
            [("SVM5", svm_only)],
            [1.0], f"W3C-30c_svm5_C{int(C*10):02d}", verbose=False)
        print(f"  SVM5 C={C}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        if best_30c is None or m['cv_log_loss_mean'] < best_30c['cv_log_loss_mean']:
            best_30c = m
    print(f"Best W3C-30c: {best_30c['cv_log_loss_mean']:.4f}")

    # ── W3C-30d: Feature subset for SVM at alpha=0.70 ────────────────────────
    print("\n--- W3C-30d: SVM feature subset sweep at alpha=0.70, C=0.35 ---")
    svm_base = lambda: SVC(C=0.35, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    best_30d = None
    # Note: KDE always uses elo_diff (col 0); SVM uses the full X matrix cols
    feat_sets = [
        (['elo_diff', 'host_advantage'], "2feat"),
        (['elo_diff', 'host_advantage', 'rank_diff'], "3feat_rk"),
        (['elo_diff', 'host_advantage', 'mv_top11_diff'], "3feat_mv"),
        (['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff'], "4feat"),
        (['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff'], "5feat"),
    ]
    results_d = []
    for feats, tag in feat_sets:
        X_f = df[feats].values
        m = cv_blend(X_f, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM", svm_base)],
            [0.70, 0.30], f"W3C-30d_{tag}", verbose=False)
        print(f"  {tag} ({len(feats)} feats): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        results_d.append(m)
        if best_30d is None or m['cv_log_loss_mean'] < best_30d['cv_log_loss_mean']:
            best_30d = m
    print(f"Best W3C-30d: {best_30d['cv_log_loss_mean']:.4f} → {best_30d['experiment']}")

    # ── W3C-30e: Even more features for SVM (7-feat or more) ─────────────────
    print("\n--- W3C-30e: 7-feat and 9-feat SVM at alpha=0.70, C=0.35 ---")
    best_30e = None
    for feats, tag in [
        (['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff', 'caps_diff', 'age_diff'], "7feat"),
        (['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff',
          'home_squad_mean_age', 'away_squad_mean_age', 'home_squad_mean_caps', 'away_squad_mean_caps'], "9feat"),
    ]:
        X_f = df[feats].values
        m = cv_blend(X_f, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM", svm_base)],
            [0.70, 0.30], f"W3C-30e_{tag}", verbose=False)
        print(f"  {tag}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (p={m['wilcoxon_vs_baseline_pvalue']:.4f})")
        if best_30e is None or m['cv_log_loss_mean'] < best_30e['cv_log_loss_mean']:
            best_30e = m
    print(f"Best W3C-30e: {best_30e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-30 experiments done.")
    all_bests = [best_30a, best_30b, best_30c, best_30d, best_30e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-30 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")

    import json
    results = sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])
    with open("/tmp/w3c30_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "note": "W3C-29a found 0.7822 by giving SVM 5 features instead of 2",
                   "all_bests": results}, f, indent=2)
    print("Summary saved to /tmp/w3c30_summary.json")


if __name__ == "__main__":
    run_experiments()
