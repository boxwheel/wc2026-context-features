"""
W3C-35: BREAKTHROUGH — rest_diff + venue_elevation gives 0.7570 (beats W2 frontier 0.7608).
Drill down on this discovery:
  a) Joint alpha×C grid with 7-feat (5+rest_diff+venue_elevation)
  b) rest_diff alone: full alpha×C grid
  c) venue_elevation alone: full alpha×C grid
  d) Different context feature combos with alpha×C sweep
  e) Replace venue_elevation with alt_diff_home/away instead
  f) Check if rest_diff + venue_elevation + home_host_home_venue adds more
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
RECORD_LOSS = 0.7570  # W3C-34b: rest_diff + venue_elevation at alpha=0.44, C=1.7


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
           "delta_vs_record_0.7570": round(ml - RECORD_LOSS, 4),
           "wilcoxon_vs_baseline_pvalue": round(pb, 4), "wilcoxon_vs_frontier_pvalue": round(pf, 4),
           "verdict_vs_baseline": v, "label_classes": list(classes),
           "blend_weights": blend_weights, "blend_components": [nm for nm, _ in components]}
    if verbose:
        print(f"\n{'='*60}\nExp: {exp_name}\nlog-loss: {ml:.4f} ± {sl:.4f}\nΔ base: {db:+.4f} Δfrontier: {ml-FRONTIER_LOSS:+.4f} (p_front={pf:.4f}) → {v}")
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
    X6_rest = np.column_stack([X5, df['rest_diff'].values.astype(float)])
    X7_re = np.column_stack([X5, df['rest_diff'].values.astype(float),
                              df['venue_elevation'].values.astype(float)])
    kde_fn = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)

    # ── W3C-35a: Joint alpha×C grid for 7-feat (5+rest_diff+venue_elevation) ──
    print("\n--- W3C-35a: alpha×C grid for 7-feat (5+rest_diff+elev) ---")
    best_35a = None
    for alpha in [0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50, 0.52]:
        for C in [1.0, 1.4, 1.7, 2.0, 2.5, 3.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X7_re, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM7", svm_fn)],
                [alpha, 1 - alpha], f"W3C-35a_a{int(alpha*100)}_C{int(C*10):02d}", verbose=False)
            if best_35a is None or m['cv_log_loss_mean'] < best_35a['cv_log_loss_mean']:
                best_35a = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD: alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} (Δfrontier={m['delta_vs_frontier_0.7608']:+.4f})")
    print(f"Best W3C-35a: {best_35a['cv_log_loss_mean']:.4f} → {best_35a['experiment']} (Δfrontier={best_35a['delta_vs_frontier_0.7608']:+.4f} p={best_35a['wilcoxon_vs_frontier_pvalue']:.4f})")

    # ── W3C-35b: rest_diff alone: full alpha×C grid ──────────────────────────
    print("\n--- W3C-35b: rest_diff alone alpha×C grid ---")
    best_35b = None
    for alpha in [0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]:
        for C in [1.0, 1.4, 1.7, 2.0, 2.5, 3.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X6_rest, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM6", svm_fn)],
                [alpha, 1 - alpha], f"W3C-35b_a{int(alpha*100)}_C{int(C*10):02d}", verbose=False)
            if best_35b is None or m['cv_log_loss_mean'] < best_35b['cv_log_loss_mean']:
                best_35b = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD: alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f}")
    print(f"Best W3C-35b: {best_35b['cv_log_loss_mean']:.4f} → {best_35b['experiment']}")

    # ── W3C-35c: venue_elevation alone: full alpha×C grid ────────────────────
    print("\n--- W3C-35c: venue_elevation alone alpha×C grid ---")
    X6_elev = np.column_stack([X5, df['venue_elevation'].values.astype(float)])
    best_35c = None
    for alpha in [0.38, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]:
        for C in [1.0, 1.4, 1.7, 2.0, 2.5, 3.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X6_elev, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM6e", svm_fn)],
                [alpha, 1 - alpha], f"W3C-35c_a{int(alpha*100)}_C{int(C*10):02d}", verbose=False)
            if best_35c is None or m['cv_log_loss_mean'] < best_35c['cv_log_loss_mean']:
                best_35c = m
    print(f"Best W3C-35c: {best_35c['cv_log_loss_mean']:.4f} → {best_35c['experiment']}")

    # ── W3C-35d: rest_diff + alt_diff_away variant ────────────────────────────
    print("\n--- W3C-35d: rest_diff + altitude variants at best alpha=0.44, C=1.7 ---")
    best_35d = None
    alt_variants = [
        ('alt_diff_away', 'altd_away'),
        ('alt_diff_home', 'altd_home'),
        ('venue_elevation', 'elev'),
    ]
    for feat, tag in alt_variants:
        X7_var = np.column_stack([X5, df['rest_diff'].values.astype(float),
                                   df[feat].values.astype(float)])
        for alpha in [0.42, 0.44, 0.46]:
            for C in [1.4, 1.7, 2.0]:
                svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
                m = cv_blend(X7_var, yenc, classes,
                    [("KDE_TS", kde_fn), ("SVM7", svm_fn)],
                    [alpha, 1 - alpha], f"W3C-35d_{tag}_a{int(alpha*100)}_C{int(C*10):02d}", verbose=False)
                if best_35d is None or m['cv_log_loss_mean'] < best_35d['cv_log_loss_mean']:
                    best_35d = m
                if m['cv_log_loss_mean'] < RECORD_LOSS:
                    print(f"  *** NEW RECORD: {tag} a={alpha} C={C} → {m['cv_log_loss_mean']:.4f}")
    print(f"Best W3C-35d: {best_35d['cv_log_loss_mean']:.4f} → {best_35d['experiment']}")

    # ── W3C-35e: rest_diff + venue_elevation + home_host_home_venue ──────────
    print("\n--- W3C-35e: 8-feat (5+rest+elev+hhv) alpha×C sweep ---")
    X8 = np.column_stack([X5, df['rest_diff'].values.astype(float),
                           df['venue_elevation'].values.astype(float),
                           df['home_host_home_venue'].values.astype(float)])
    best_35e = None
    for alpha in [0.40, 0.42, 0.44, 0.46]:
        for C in [1.4, 1.7, 2.0, 2.5]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X8, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM8", svm_fn)],
                [alpha, 1 - alpha], f"W3C-35e_a{int(alpha*100)}_C{int(C*10):02d}", verbose=False)
            if best_35e is None or m['cv_log_loss_mean'] < best_35e['cv_log_loss_mean']:
                best_35e = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD: 8-feat a={alpha} C={C} → {m['cv_log_loss_mean']:.4f}")
    print(f"Best W3C-35e: {best_35e['cv_log_loss_mean']:.4f} → {best_35e['experiment']}")

    # ── W3C-35f: rest_days_away alone (not diff) — test raw away fatigue ─────
    print("\n--- W3C-35f: rest_days_away alone (raw away fatigue) ---")
    X6_raw = np.column_stack([X5, df['rest_days_away'].values.astype(float)])
    best_35f = None
    for alpha in [0.40, 0.42, 0.44, 0.46]:
        for C in [1.4, 1.7, 2.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X6_raw, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM6r", svm_fn)],
                [alpha, 1 - alpha], f"W3C-35f_a{int(alpha*100)}_C{int(C*10):02d}", verbose=False)
            if best_35f is None or m['cv_log_loss_mean'] < best_35f['cv_log_loss_mean']:
                best_35f = m
    print(f"Best W3C-35f: {best_35f['cv_log_loss_mean']:.4f} → {best_35f['experiment']}")

    print("\nAll W3C-35 experiments done.")
    all_bests = [x for x in [best_35a, best_35b, best_35c, best_35d, best_35e, best_35f] if x is not None]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-35 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")
    print(f"Δ vs frontier (0.7608): {overall['delta_vs_frontier_0.7608']:+.4f} (p_front={overall['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Δ vs record  (0.7570): {overall['delta_vs_record_0.7570']:+.4f}")

    with open("/tmp/w3c35_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "frontier": FRONTIER_LOSS,
                   "previous_record": RECORD_LOSS,
                   "note": "Context feature grid search: rest_diff + venue_elevation",
                   "all_bests": sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])}, f, indent=2)
    print("Summary saved to /tmp/w3c35_summary.json")


if __name__ == "__main__":
    run_experiments()
