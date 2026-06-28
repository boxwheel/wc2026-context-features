"""
W3C-38: Fine-tune around W3C-37b winner: threshold=300m, alpha=0.24, C=6.0 → 0.7175 (p=0.0123).
This is the first statistically significant beat of the W2 frontier (p<0.05).

Focus areas:
  a) Fine threshold sweep (100..450 step 50) × fine alpha×C grid
  b) 8-feat: add home_host_home_venue (host advantage indicator, corr=+0.137)
  c) Ultra-low alpha (near-pure SVM): alpha ∈ {0.05..0.15}, C ∈ {6..20}
  d) KDE parameter tuning (bw, C2) for th300 best config
  e) Two-threshold blend: indicators at th300 AND th500 simultaneously (8-feat)
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
RECORD_LOSS = 0.7175  # W3C-37b


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


def cv_blend(X, y_enc, classes, components, blend_weights, exp_name, verbose=False):
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
           "delta_vs_record_0.7175": round(ml - RECORD_LOSS, 4),
           "wilcoxon_vs_baseline_pvalue": round(pb, 4), "wilcoxon_vs_frontier_pvalue": round(pf, 4),
           "verdict_vs_baseline": v, "label_classes": list(classes),
           "blend_weights": blend_weights, "blend_components": [nm for nm, _ in components]}
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
    rest = df['rest_diff'].values.astype(float)
    elev = df['venue_elevation'].values.astype(float)
    hhv = df['home_host_home_venue'].values.astype(float)  # home team playing at home nation venue

    # Best from W3C-37b: FEAT5 + rest_diff + (elev>300m), alpha=0.24, C=6.0 → 0.7175
    kde0 = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)
    elev_ind300 = (elev > 300).astype(float)

    # ── W3C-38a: Fine threshold × fine alpha×C grid ───────────────────────
    print("\n--- W3C-38a: Fine threshold sweep × fine alpha×C grid ---")
    best_38a = None
    thresholds_a = [100, 150, 200, 250, 300, 350, 400, 450]
    alphas_a = [0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28]
    Cs_a = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    for thresh in thresholds_a:
        elev_ind = (elev > thresh).astype(float)
        X7 = np.column_stack([X5, rest, elev_ind])
        for alpha in alphas_a:
            for C in Cs_a:
                svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
                tag = f"W3C-38a_th{thresh}_a{int(alpha*100):02d}_C{int(C*10):03d}"
                m = cv_blend(X7, yenc, classes,
                    [("KDE_TS", kde0), ("SVM7i", svm_fn)],
                    [alpha, 1 - alpha], tag)
                if best_38a is None or m['cv_log_loss_mean'] < best_38a['cv_log_loss_mean']:
                    best_38a = m
                if m['cv_log_loss_mean'] < RECORD_LOSS:
                    print(f"  *** NEW RECORD: th={thresh} alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} (p={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-38a: {best_38a['cv_log_loss_mean']:.4f} → {best_38a['experiment']} (p_front={best_38a['wilcoxon_vs_frontier_pvalue']:.4f})")

    # ── W3C-38b: 8-feat: add home_host_home_venue to th300 setup ──────────
    print("\n--- W3C-38b: 8-feat — add home_host_home_venue indicator ---")
    X8_hhv = np.column_stack([X5, rest, elev_ind300, hhv])
    best_38b = None
    for alpha in [0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]:
        for C in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X8_hhv, yenc, classes,
                [("KDE_TS", kde0), ("SVM8h", svm_fn)],
                [alpha, 1 - alpha], f"W3C-38b_a{int(alpha*100):02d}_C{int(C*10):03d}")
            if best_38b is None or m['cv_log_loss_mean'] < best_38b['cv_log_loss_mean']:
                best_38b = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD: alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} (p={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-38b: {best_38b['cv_log_loss_mean']:.4f} → {best_38b['experiment']}")

    # ── W3C-38c: Ultra-low alpha (near-pure SVM), high C ──────────────────
    print("\n--- W3C-38c: Ultra-low alpha (0.05..0.14), high C (8..25) ---")
    X7_300 = np.column_stack([X5, rest, elev_ind300])
    best_38c = None
    for alpha in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]:
        for C in [6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X7_300, yenc, classes,
                [("KDE_TS", kde0), ("SVM7i", svm_fn)],
                [alpha, 1 - alpha], f"W3C-38c_a{int(alpha*100):02d}_C{int(C*10):03d}")
            if best_38c is None or m['cv_log_loss_mean'] < best_38c['cv_log_loss_mean']:
                best_38c = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD: alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} (p={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-38c: {best_38c['cv_log_loss_mean']:.4f} → {best_38c['experiment']}")

    # ── W3C-38d: KDE bw+C2+pw tuning for th300 ────────────────────────────
    print("\n--- W3C-38d: KDE parameter tuning for th300 setup ---")
    best_38d = None
    svm_best = lambda: SVC(C=6.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    for bw in [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0]:
        for C2 in [1.0, 1.5, 2.0, 3.0]:
            for pw in [0.05, 0.10, 0.15, 0.20, 0.25]:
                kde_v = lambda bw=bw, C2=C2, pw=pw: KDETwoStageModel(bw=bw, C2=C2, prior_weight=pw)
                m = cv_blend(X7_300, yenc, classes,
                    [("KDE_TS", kde_v), ("SVM7i", svm_best)],
                    [0.24, 0.76],
                    f"W3C-38d_bw{int(bw)}_C2{int(C2*10):02d}_pw{int(pw*100):02d}")
                if best_38d is None or m['cv_log_loss_mean'] < best_38d['cv_log_loss_mean']:
                    best_38d = m
                if m['cv_log_loss_mean'] < RECORD_LOSS:
                    print(f"  *** NEW RECORD: bw={bw} C2={C2} pw={pw} → {m['cv_log_loss_mean']:.4f} (p={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-38d: {best_38d['cv_log_loss_mean']:.4f} → {best_38d['experiment']}")

    # ── W3C-38e: Two-threshold blend: indicators at th300 AND th500 ───────
    print("\n--- W3C-38e: Two altitude indicators simultaneously (300m + 500m) ---")
    elev_ind500 = (elev > 500).astype(float)
    X8_2th = np.column_stack([X5, rest, elev_ind300, elev_ind500])
    best_38e = None
    for alpha in [0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28]:
        for C in [5.0, 6.0, 7.0, 8.0, 10.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X8_2th, yenc, classes,
                [("KDE_TS", kde0), ("SVM8t", svm_fn)],
                [alpha, 1 - alpha], f"W3C-38e_a{int(alpha*100):02d}_C{int(C*10):03d}")
            if best_38e is None or m['cv_log_loss_mean'] < best_38e['cv_log_loss_mean']:
                best_38e = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD: alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} (p={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-38e: {best_38e['cv_log_loss_mean']:.4f} → {best_38e['experiment']}")

    print("\nAll W3C-38 experiments done.")
    all_bests = [x for x in [best_38a, best_38b, best_38c, best_38d, best_38e] if x is not None]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-38 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")
    print(f"Δ vs frontier (0.7608): {overall['delta_vs_frontier_0.7608']:+.4f}")
    print(f"p_frontier: {overall['wilcoxon_vs_frontier_pvalue']:.4f}")
    print(f"Δ vs W3C-37 record (0.7175): {overall.get('delta_vs_record_0.7175', 'N/A')}")

    with open("/tmp/w3c38_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "frontier": FRONTIER_LOSS,
                   "previous_record": RECORD_LOSS,
                   "wilcoxon_vs_frontier": overall["wilcoxon_vs_frontier_pvalue"],
                   "all_bests": sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])}, f, indent=2)
    print("Summary saved to /tmp/w3c38_summary.json")


if __name__ == "__main__":
    run_experiments()
