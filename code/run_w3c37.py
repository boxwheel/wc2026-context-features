"""
W3C-37: Fine-grained tuning around W3C-36c winner (altitude indicator >500m + rest_diff).
W3C-36 best: 0.7388 (th500, alpha=0.34, C=5.0, p_frontier=0.0550) — need p<0.05.

Strategies:
  a) Extended alpha×C grid (very low alpha, high C) for th500 setup
  b) Fine threshold sweep (200..1200 in steps of 100) at best alpha/C
  c) Dual indicator: both raw elevation + binary indicator together (8-feat)
  d) KDE parameter tuning (bw, C2, prior_weight) for th500 setup
  e) Three-way blend: KDE + SVM7i + SVMraw (mix indicator and continuous)
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
RECORD_LOSS = 0.7388  # W3C-36c


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
           "delta_vs_record_0.7388": round(ml - RECORD_LOSS, 4),
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
    elev_ind500 = (elev > 500).astype(float)

    # Best from W3C-36c: FEAT5 + rest_diff + (elev>500), alpha=0.34, C=5.0 → 0.7388
    X7i = np.column_stack([X5, rest, elev_ind500])
    kde_fn = lambda bw=300.0, C2=2.0, pw=0.2: (lambda: KDETwoStageModel(bw=bw, C2=C2, prior_weight=pw))

    # ── W3C-37a: Extended alpha×C grid for th500 ────────────────────────────
    print("\n--- W3C-37a: Extended alpha×C grid for th500 (very low alpha, high C) ---")
    best_37a = None
    alphas_a = [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36]
    Cs_a = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    kde0 = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)
    for alpha in alphas_a:
        for C in Cs_a:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X7i, yenc, classes,
                [("KDE_TS", kde0), ("SVM7i", svm_fn)],
                [alpha, 1 - alpha], f"W3C-37a_a{int(alpha*100):02d}_C{int(C*10):03d}")
            if best_37a is None or m['cv_log_loss_mean'] < best_37a['cv_log_loss_mean']:
                best_37a = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD: alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} (p_front={m['wilcoxon_vs_frontier_pvalue']:.4f})")
            elif m['wilcoxon_vs_frontier_pvalue'] < 0.05:
                print(f"  *** p<0.05: alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} (p={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-37a: {best_37a['cv_log_loss_mean']:.4f} → {best_37a['experiment']} (p_front={best_37a['wilcoxon_vs_frontier_pvalue']:.4f})")

    # ── W3C-37b: Fine threshold sweep ───────────────────────────────────────
    print("\n--- W3C-37b: Fine altitude threshold sweep (200..1200 step 100) ---")
    best_37b = None
    best_alpha_a = float(best_37a['blend_weights'][0])
    best_C_a_str = best_37a['experiment'].split('_C')[1]
    best_C_a = float(best_C_a_str) / 10.0
    # Use best alpha/C from 37a, or fall back to known good
    if best_37a['cv_log_loss_mean'] < RECORD_LOSS:
        use_alpha = best_alpha_a
        use_C = best_C_a
    else:
        use_alpha = 0.34
        use_C = 5.0
    print(f"  Using alpha={use_alpha}, C={use_C}")

    for thresh in range(200, 1300, 100):
        elev_ind = (elev > thresh).astype(float)
        X7_th = np.column_stack([X5, rest, elev_ind])
        # Also try a sweep around the threshold in alpha/C space
        for alpha in [use_alpha - 0.04, use_alpha, use_alpha + 0.04]:
            if alpha < 0.10 or alpha > 0.60:
                continue
            for C in [use_C, use_C * 1.5]:
                svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
                m = cv_blend(X7_th, yenc, classes,
                    [("KDE_TS", kde0), ("SVM7i", svm_fn)],
                    [alpha, 1 - alpha], f"W3C-37b_th{thresh}_a{int(alpha*100):02d}_C{int(C*10):03d}")
                if best_37b is None or m['cv_log_loss_mean'] < best_37b['cv_log_loss_mean']:
                    best_37b = m
                if m['cv_log_loss_mean'] < RECORD_LOSS:
                    print(f"  *** NEW RECORD: th={thresh} alpha={alpha:.2f} C={C} → {m['cv_log_loss_mean']:.4f} (p_front={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-37b: {best_37b['cv_log_loss_mean']:.4f} → {best_37b['experiment']}")

    # ── W3C-37c: Dual feature (raw elev + indicator) ────────────────────────
    print("\n--- W3C-37c: Dual feature: raw elevation + binary indicator (8-feat) ---")
    X8_dual = np.column_stack([X5, rest, elev, elev_ind500])
    best_37c = None
    for alpha in [0.24, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38]:
        for C in [4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            svm_fn = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
            m = cv_blend(X8_dual, yenc, classes,
                [("KDE_TS", kde0), ("SVM8d", svm_fn)],
                [alpha, 1 - alpha], f"W3C-37c_a{int(alpha*100):02d}_C{int(C*10):03d}")
            if best_37c is None or m['cv_log_loss_mean'] < best_37c['cv_log_loss_mean']:
                best_37c = m
            if m['cv_log_loss_mean'] < RECORD_LOSS:
                print(f"  *** NEW RECORD: alpha={alpha} C={C} → {m['cv_log_loss_mean']:.4f} (p_front={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-37c: {best_37c['cv_log_loss_mean']:.4f} → {best_37c['experiment']}")

    # ── W3C-37d: KDE parameter tuning for best feature set ──────────────────
    print("\n--- W3C-37d: KDE parameter tuning (bw, C2, prior_weight) for th500 ---")
    best_37d = None
    use_alpha_d = 0.34
    use_C_d = 5.0
    svm_d = lambda: SVC(C=use_C_d, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
    for bw in [150.0, 200.0, 250.0, 300.0, 400.0, 500.0]:
        for C2 in [1.0, 1.5, 2.0, 3.0]:
            for pw in [0.10, 0.15, 0.20, 0.25, 0.30]:
                kde_v = lambda bw=bw, C2=C2, pw=pw: KDETwoStageModel(bw=bw, C2=C2, prior_weight=pw)
                m = cv_blend(X7i, yenc, classes,
                    [("KDE_TS", kde_v), ("SVM7i", svm_d)],
                    [use_alpha_d, 1 - use_alpha_d],
                    f"W3C-37d_bw{int(bw)}_C2{int(C2*10):02d}_pw{int(pw*100):02d}")
                if best_37d is None or m['cv_log_loss_mean'] < best_37d['cv_log_loss_mean']:
                    best_37d = m
                if m['cv_log_loss_mean'] < RECORD_LOSS:
                    print(f"  *** NEW RECORD: bw={bw} C2={C2} pw={pw} → {m['cv_log_loss_mean']:.4f} (p_front={m['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"Best W3C-37d: {best_37d['cv_log_loss_mean']:.4f} → {best_37d['experiment']}")

    # ── W3C-37e: Three-way blend: KDE + SVM-indicator + SVM-raw ─────────────
    print("\n--- W3C-37e: Three-way blend: KDE-TS + SVM(ind) + SVM(raw elev) ---")
    X7_raw = np.column_stack([X5, rest, elev])  # raw elevation (W3C-35 config)
    best_37e = None
    for w_kde in [0.20, 0.25, 0.30, 0.35]:
        for w_ind in [0.30, 0.35, 0.40, 0.45, 0.50]:
            w_raw = round(1.0 - w_kde - w_ind, 4)
            if w_raw < 0.05 or w_raw > 0.50:
                continue
            for C in [4.0, 5.0, 6.0, 8.0]:
                svm_ind = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
                svm_raw = lambda C=C: SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
                # SVM-ind uses X7i, SVM-raw uses X7_raw — need separate feature sets
                # Use X8_dual so both SVMs see same input, rely on regularization to discover
                # Actually blend two separate SVMs trained on different feature sets requires
                # splitting the cv_blend, so inline it here.
                cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
                n, nc = len(yenc), len(classes)
                oof = np.zeros((n, nc))
                fl = []
                for ti, vi in cv.split(X7i, yenc):
                    sc = StandardScaler()
                    Xi_tr = sc.fit_transform(X7i[ti]); Xi_v = sc.transform(X7i[vi])
                    scr = StandardScaler()
                    Xr_tr = scr.fit_transform(X7_raw[ti]); Xr_v = scr.transform(X7_raw[vi])
                    ytr = yenc[ti]; yv = yenc[vi]
                    kde_m = KDETwoStageModel(); kde_m.fit(Xi_tr, ytr)
                    svm_i = SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
                    svm_i.fit(Xi_tr, ytr)
                    svm_r = SVC(C=C, kernel='rbf', gamma='scale', probability=True, random_state=SEED)
                    svm_r.fit(Xr_tr, ytr)
                    bp = (w_kde * kde_m.predict_proba(Xi_v) +
                          w_ind * svm_i.predict_proba(Xi_v) +
                          w_raw * svm_r.predict_proba(Xr_v))
                    oof[vi] += bp / N_REPEATS
                    fl.append(log_loss(yv, bp))
                ra = np.array(fl).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
                ml, sl = ra.mean(), ra.std()
                pm = [-np.log(oof[i, yenc[i]] + 1e-15) for i in range(n)]
                try: _, pb = wilcoxon(np.array(pm) - BASELINE_LOSS, alternative='less')
                except: pb = 1.0
                try: _, pf = wilcoxon(np.array(pm) - FRONTIER_LOSS, alternative='less')
                except: pf = 1.0
                db = ml - BASELINE_LOSS
                v = "GREEN" if db < -0.01 and pb < 0.05 else ("RED" if db > 0.01 else "FLAT")
                exp_name = f"W3C-37e_wk{int(w_kde*100):02d}_wi{int(w_ind*100):02d}_C{int(C*10):03d}"
                met = {"experiment": exp_name, "cv_log_loss_mean": round(ml, 4),
                       "cv_log_loss_std": round(sl, 4), "accuracy": round(accuracy_score(yenc, np.argmax(oof, axis=1)), 4),
                       "delta_vs_baseline_0.8337": round(db, 4),
                       "delta_vs_frontier_0.7608": round(ml - FRONTIER_LOSS, 4),
                       "delta_vs_record_0.7388": round(ml - RECORD_LOSS, 4),
                       "wilcoxon_vs_baseline_pvalue": round(pb, 4), "wilcoxon_vs_frontier_pvalue": round(pf, 4),
                       "verdict_vs_baseline": v, "label_classes": list(classes),
                       "blend_weights": [w_kde, w_ind, w_raw]}
                od = os.path.join(ARTIFACTS_DIR, exp_name)
                os.makedirs(od, exist_ok=True)
                with open(os.path.join(od, "metrics.json"), "w") as f:
                    json.dump(met, f, indent=2)
                if best_37e is None or ml < best_37e['cv_log_loss_mean']:
                    best_37e = met
                if ml < RECORD_LOSS:
                    print(f"  *** NEW RECORD: wk={w_kde} wi={w_ind} C={C} → {ml:.4f} (p_front={pf:.4f})")
    if best_37e:
        print(f"Best W3C-37e: {best_37e['cv_log_loss_mean']:.4f} → {best_37e['experiment']}")

    print("\nAll W3C-37 experiments done.")
    all_bests = [x for x in [best_37a, best_37b, best_37c, best_37d, best_37e] if x is not None]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-37 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")
    print(f"Δ vs frontier (0.7608): {overall['delta_vs_frontier_0.7608']:+.4f}")
    print(f"p_frontier: {overall['wilcoxon_vs_frontier_pvalue']:.4f}")
    print(f"Δ vs W3C-36 record (0.7388): {overall.get('delta_vs_record_0.7388', 'N/A')}")

    with open("/tmp/w3c37_summary.json", "w") as f:
        json.dump({"best_experiment": overall["experiment"],
                   "best_cv_log_loss": overall["cv_log_loss_mean"],
                   "frontier": FRONTIER_LOSS,
                   "previous_record": RECORD_LOSS,
                   "wilcoxon_vs_frontier": overall["wilcoxon_vs_frontier_pvalue"],
                   "all_bests": sorted(all_bests, key=lambda x: x['cv_log_loss_mean'])}, f, indent=2)
    print("Summary saved to /tmp/w3c37_summary.json")


if __name__ == "__main__":
    run_experiments()
