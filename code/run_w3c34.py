"""
W3C-34: Context features as 6th feature candidates.
W3C-33 exhausted SVM kernel variants, calibration, ensembles — no improvement over 0.7744.
Current architecture appears saturated. Try new signal sources:

Context features (available from include_context=True):
  - match_num_diff: corr=+0.277 with home_win (highest unseen feature)
  - rest_diff: corr=+0.125
  - venue_elevation: corr=+0.082
  - home_host_home_venue: corr=+0.137

  a) Single context feature additions at best config (alpha=0.44, C=1.7)
  b) Two-context-feature combos
  c) Replace weakest existing feature (host_advantage) with match_num_diff
  d) Replace gk_mv_diff with match_num_diff (ablation)
  e) Full 8-feature set (5 existing + 3 context)
  f) Alpha sweep for best context config
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
from features import build_match_features

ARTIFACTS_DIR = "/home/user/research/wave3-context/artifacts"
SEED = 0
N_SPLITS = 5
N_REPEATS = 10
BASELINE_LOSS = 0.8337
FRONTIER_LOSS = 0.7608
RECORD_LOSS = 0.7744


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
    print("Loading WC-2026 features (with context)...")
    df = build_match_features(include_context=True)
    y = df["label"].values
    le = LabelEncoder()
    yenc = le.fit_transform(y)
    classes = le.classes_
    print(f"n={len(yenc)}, classes={classes}")

    FEAT5 = ['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff']
    X5 = df[FEAT5].values
    kde_fn = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)

    # Best configs found so far
    ALPHA_BEST = 0.44
    C_BEST = 1.7

    def svm(C=C_BEST):
        return SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)

    # ── W3C-34a: Single context feature additions ─────────────────────────────
    print("\n--- W3C-34a: Single context feature additions at (alpha=0.44, C=1.7) ---")
    ctx_feats = [
        ('match_num_diff', '+match_num_diff'),
        ('rest_diff', '+rest_diff'),
        ('venue_elevation', '+venue_elev'),
        ('home_host_home_venue', '+home_host_venue'),
        ('away_host_home_venue', '+away_host_venue'),
        ('alt_diff_away', '+alt_diff_away'),
        ('travel_diff', '+travel_diff'),
        ('kickoff_local_hour', '+kickoff_hour'),
    ]
    best_34a = None
    results_34a = []
    for feat, tag in ctx_feats:
        try:
            X6 = np.column_stack([X5, df[feat].values.astype(float)])
            svm6 = lambda: svm(C=C_BEST)
            m = cv_blend(X6, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM6", svm6)],
                [ALPHA_BEST, 1 - ALPHA_BEST], f"W3C-34a_{tag}", verbose=False)
            print(f"  {tag}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
            results_34a.append(m)
            if best_34a is None or m['cv_log_loss_mean'] < best_34a['cv_log_loss_mean']:
                best_34a = m
        except Exception as e:
            print(f"  {tag}: ERROR {e}")
    print(f"Best W3C-34a: {best_34a['cv_log_loss_mean']:.4f}")

    # ── W3C-34b: Two context features simultaneously ──────────────────────────
    print("\n--- W3C-34b: Two-context-feature combos ---")
    best_34b = None
    ctx_combos = [
        (['match_num_diff', 'rest_diff'], 'mnd_restd'),
        (['match_num_diff', 'venue_elevation'], 'mnd_elev'),
        (['match_num_diff', 'home_host_home_venue'], 'mnd_hhv'),
        (['rest_diff', 'venue_elevation'], 'restd_elev'),
        (['match_num_diff', 'alt_diff_away'], 'mnd_altd'),
    ]
    for feats, tag in ctx_combos:
        try:
            X7 = np.column_stack([X5] + [df[f].values.astype(float) for f in feats])
            svm7 = lambda: svm(C=C_BEST)
            m = cv_blend(X7, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM7", svm7)],
                [ALPHA_BEST, 1 - ALPHA_BEST], f"W3C-34b_{tag}", verbose=False)
            print(f"  {tag}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
            if best_34b is None or m['cv_log_loss_mean'] < best_34b['cv_log_loss_mean']:
                best_34b = m
        except Exception as e:
            print(f"  {tag}: ERROR {e}")
    if best_34b:
        print(f"Best W3C-34b: {best_34b['cv_log_loss_mean']:.4f}")

    # ── W3C-34c: Replace weakest feature with match_num_diff ─────────────────
    print("\n--- W3C-34c: Feature replacement ablation with match_num_diff ---")
    best_34c = None
    mnd = df['match_num_diff'].values.astype(float)
    feat_ablations = [
        (['match_num_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff'], 'drop_elo'),
        (['elo_diff', 'match_num_diff', 'rank_diff', 'mv_top11_diff', 'gk_mv_diff'], 'drop_host'),
        (['elo_diff', 'host_advantage', 'match_num_diff', 'mv_top11_diff', 'gk_mv_diff'], 'drop_rank'),
        (['elo_diff', 'host_advantage', 'rank_diff', 'match_num_diff', 'gk_mv_diff'], 'drop_mv'),
        (['elo_diff', 'host_advantage', 'rank_diff', 'mv_top11_diff', 'match_num_diff'], 'drop_gk'),
    ]
    for feats, tag in feat_ablations:
        X_ab = df[feats].values.astype(float)
        svm_ab = lambda: svm(C=C_BEST)
        m = cv_blend(X_ab, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM_ab", svm_ab)],
            [ALPHA_BEST, 1 - ALPHA_BEST], f"W3C-34c_{tag}", verbose=False)
        print(f"  {tag} (feats: {feats[:2]}...): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
        if best_34c is None or m['cv_log_loss_mean'] < best_34c['cv_log_loss_mean']:
            best_34c = m
    print(f"Best W3C-34c: {best_34c['cv_log_loss_mean']:.4f}")

    # ── W3C-34d: Alpha sweep for best single context feature ─────────────────
    print("\n--- W3C-34d: Alpha sweep for +match_num_diff (best context feat if it wins) ---")
    best_34d = None
    X6_mnd = np.column_stack([X5, df['match_num_diff'].values.astype(float)])
    for alpha in [0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]:
        for C in [1.7, 2.0, 2.5]:
            svm6 = lambda C=C: svm(C=C)
            m = cv_blend(X6_mnd, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM6", svm6)],
                [alpha, 1 - alpha], f"W3C-34d_mnd_a{int(alpha*100)}_C{int(C*10):02d}", verbose=False)
            if best_34d is None or m['cv_log_loss_mean'] < best_34d['cv_log_loss_mean']:
                best_34d = m
            if m['cv_log_loss_mean'] <= RECORD_LOSS:
                print(f"  NEW RECORD: alpha={alpha} C={C}: {m['cv_log_loss_mean']:.4f}")
    print(f"Best W3C-34d: {best_34d['cv_log_loss_mean']:.4f} → {best_34d['experiment']}")

    # ── W3C-34e: Full context feature exploration ─────────────────────────────
    print("\n--- W3C-34e: Larger context feature sets ---")
    best_34e = None
    context_cols = ['rest_diff', 'venue_elevation', 'home_host_home_venue', 'match_num_diff']
    for n_ctx in [1, 2, 3, 4]:
        X_ctx = np.column_stack([X5] + [df[c].values.astype(float) for c in context_cols[:n_ctx]])
        svm_ctx = lambda: svm(C=C_BEST)
        m = cv_blend(X_ctx, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM_ctx", svm_ctx)],
            [ALPHA_BEST, 1 - ALPHA_BEST], f"W3C-34e_{n_ctx}ctx", verbose=False)
        print(f"  {n_ctx}ctx ({context_cols[:n_ctx]}): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']} (Δrec={m['delta_vs_record_0.7744']:+.4f})")
        if best_34e is None or m['cv_log_loss_mean'] < best_34e['cv_log_loss_mean']:
            best_34e = m
    print(f"Best W3C-34e: {best_34e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-34 experiments done.")
    all_bests = [x for x in [best_34a, best_34b, best_34c, best_34d, best_34e] if x is not None]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-34 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")
    print(f"Previous record: {RECORD_LOSS:.4f}, delta: {overall['cv_log_loss_mean'] - RECORD_LOSS:+.4f}")

    with open("/tmp/w3c34_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "previous_record": RECORD_LOSS,
                   "note": "Context features as 6th+ feature candidates",
                   "all_bests": sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])}, f, indent=2)
    print("Summary saved to /tmp/w3c34_summary.json")


if __name__ == "__main__":
    run_experiments()
