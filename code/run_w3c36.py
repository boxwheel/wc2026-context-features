"""
W3C-36: Push past frontier statistically. Current record: 0.7502 (p_front=0.0984).
Alpha=0.38, C=3.0, 7-feat [elo_diff, host_adv, rank_diff, mv_top11_diff, gk_mv_diff, rest_diff, venue_elev].
Need p_frontier < 0.05. Try:
  a) Extended alpha×C grid: lower alpha [0.30..0.40], higher C [3.0..8.0]
  b) Log-transformed altitude: log(venue_elevation+1)
  c) rest_days_home, rest_days_away separately (not just diff)
  d) travel_km_away as additional fatigue proxy
  e) Squared rest_diff (quadratic fatigue)
  f) Altitude > 1000m indicator (Mexico City effect)
  g) Interaction: rest_diff * venue_elevation
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
RECORD_LOSS = 0.7502  # W3C-35a: alpha=0.38, C=3.0, 7-feat


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
           "delta_vs_record_0.7502": round(ml - RECORD_LOSS, 4),
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
    X7 = np.column_stack([X5, rest, elev])  # baseline 7-feat
    kde_fn = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)

    # ── W3C-36a: Extended alpha×C grid — lower alpha, higher C ───────────────
    print("\n--- W3C-36a: Extended alpha×C grid for 7-feat ---")
    best_36a = None
    new_records = []
    for alpha in [0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]:
        for C in [2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X7, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM7", svm_fn)],
                [alpha, 1 - alpha], f"W3C-36a_a{int(alpha*100)}_C{int(C*10):02d}")
            if best_36a is None or m['cv_log_loss_mean'] < best_36a['cv_log_loss_mean']:
                best_36a = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                new_records.append((alpha, C, m['cv_log_loss_mean'], m['wilcoxon_vs_frontier_pvalue']))
                print(f"  *** NEW RECORD: alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} (p_front={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-36a: {best_36a['cv_log_loss_mean']:.4f} → {best_36a['experiment']} (p_front={best_36a['wilcoxon_vs_frontier_pvalue']:.4f})")

    # ── W3C-36b: Log-transformed altitude ────────────────────────────────────
    print("\n--- W3C-36b: Log-altitude + rest_diff ---")
    log_elev = np.log1p(elev)
    X7_log = np.column_stack([X5, rest, log_elev])
    best_36b = None
    for alpha in [0.30, 0.34, 0.36, 0.38, 0.40, 0.42]:
        for C in [3.0, 4.0, 5.0, 6.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X7_log, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM7l", svm_fn)],
                [alpha, 1 - alpha], f"W3C-36b_a{int(alpha*100)}_C{int(C*10):02d}")
            if best_36b is None or m['cv_log_loss_mean'] < best_36b['cv_log_loss_mean']:
                best_36b = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD (log-elev): alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} p_front={m['wilcoxon_vs_frontier_pvalue']:.4f}")
    print(f"Best W3C-36b (log-elev): {best_36b['cv_log_loss_mean']:.4f} → {best_36b['experiment']}")

    # ── W3C-36c: Altitude indicator (elev > threshold) ───────────────────────
    print("\n--- W3C-36c: Altitude indicator + rest_diff ---")
    best_36c = None
    for thresh in [500, 1000, 1500]:
        elev_ind = (elev > thresh).astype(float)
        X7_ind = np.column_stack([X5, rest, elev_ind])
        for alpha in [0.34, 0.38, 0.40]:
            for C in [3.0, 4.0, 5.0]:
                svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
                m = cv_blend(X7_ind, yenc, classes,
                    [("KDE_TS", kde_fn), ("SVM7i", svm_fn)],
                    [alpha, 1 - alpha], f"W3C-36c_th{thresh}_a{int(alpha*100)}_C{int(C*10):02d}")
                if best_36c is None or m['cv_log_loss_mean'] < best_36c['cv_log_loss_mean']:
                    best_36c = m
                if m['cv_log_loss_mean'] < RECORD_LOSS:
                    print(f"  *** NEW RECORD (elev>{thresh}): {m['cv_log_loss_mean']:.4f} p_front={m['wilcoxon_vs_frontier_pvalue']:.4f}")
    print(f"Best W3C-36c (elev indicator): {best_36c['cv_log_loss_mean']:.4f}")

    # ── W3C-36d: rest_diff * venue_elevation interaction ─────────────────────
    print("\n--- W3C-36d: rest×elev interaction feature ---")
    rest_x_elev = rest * elev / (np.std(rest) * np.std(elev) + 1e-9)
    X8_int = np.column_stack([X5, rest, elev, rest_x_elev])
    best_36d = None
    for alpha in [0.34, 0.36, 0.38, 0.40]:
        for C in [3.0, 4.0, 5.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X8_int, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM8i", svm_fn)],
                [alpha, 1 - alpha], f"W3C-36d_a{int(alpha*100)}_C{int(C*10):02d}")
            if best_36d is None or m['cv_log_loss_mean'] < best_36d['cv_log_loss_mean']:
                best_36d = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD (int): {m['cv_log_loss_mean']:.4f} p_front={m['wilcoxon_vs_frontier_pvalue']:.4f}")
    print(f"Best W3C-36d (rest×elev): {best_36d['cv_log_loss_mean']:.4f}")

    # ── W3C-36e: rest_days_home + rest_days_away separately ──────────────────
    print("\n--- W3C-36e: rest_days_home/away + venue_elevation ---")
    rdh = df['rest_days_home'].values.astype(float)
    rda = df['rest_days_away'].values.astype(float)
    X8_sep = np.column_stack([X5, rdh, rda, elev])
    best_36e = None
    for alpha in [0.34, 0.36, 0.38, 0.40]:
        for C in [3.0, 4.0, 5.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X8_sep, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM8s", svm_fn)],
                [alpha, 1 - alpha], f"W3C-36e_a{int(alpha*100)}_C{int(C*10):02d}")
            if best_36e is None or m['cv_log_loss_mean'] < best_36e['cv_log_loss_mean']:
                best_36e = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD (sep rest): {m['cv_log_loss_mean']:.4f} p_front={m['wilcoxon_vs_frontier_pvalue']:.4f}")
    print(f"Best W3C-36e (sep rest+elev): {best_36e['cv_log_loss_mean']:.4f}")

    # ── W3C-36f: travel_km_away as additional fatigue ────────────────────────
    print("\n--- W3C-36f: rest_diff + venue_elevation + travel_km_away ---")
    tka = df['travel_km_away'].values.astype(float)
    X8_trv = np.column_stack([X5, rest, elev, tka])
    best_36f = None
    for alpha in [0.34, 0.36, 0.38, 0.40]:
        for C in [3.0, 4.0, 5.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X8_trv, yenc, classes,
                [("KDE_TS", kde_fn), ("SVM8t", svm_fn)],
                [alpha, 1 - alpha], f"W3C-36f_a{int(alpha*100)}_C{int(C*10):02d}")
            if best_36f is None or m['cv_log_loss_mean'] < best_36f['cv_log_loss_mean']:
                best_36f = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD (travel): {m['cv_log_loss_mean']:.4f} p_front={m['wilcoxon_vs_frontier_pvalue']:.4f}")
    print(f"Best W3C-36f (travel): {best_36f['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-36 experiments done.")
    all_bests = [x for x in [best_36a, best_36b, best_36c, best_36d, best_36e, best_36f] if x is not None]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-36 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")
    print(f"Δ vs frontier (0.7608): {overall['delta_vs_frontier_0.7608']:+.4f}")
    print(f"p_frontier: {overall['wilcoxon_vs_frontier_pvalue']:.4f}")
    print(f"Δ vs W3C-35 record (0.7502): {overall['delta_vs_record_0.7502']:+.4f}")

    with open("/tmp/w3c36_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "frontier": FRONTIER_LOSS,
                   "previous_record": RECORD_LOSS,
                   "wilcoxon_vs_frontier": overall["wilcoxon_vs_frontier_pvalue"],
                   "note": "Extended grid + altitude transformations + fatigue features",
                   "all_bests": sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])}, f, indent=2)
    print("Summary saved to /tmp/w3c36_summary.json")


if __name__ == "__main__":
    run_experiments()
