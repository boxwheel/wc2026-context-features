"""
W3C-19: Feature-expanded SVM paired with KDE-TS.
Key idea: KDE-TS handles non-monotone draw probability from elo_diff.
SVM uses richer features (squad market value, caps, age, rank, context).
  a) KDE-TS(elo) + SVM(all 16 features)
  b) KDE-TS(elo) + SVM(squad+elo features, 8 features)
  c) KDE-TS(elo) + LogReg(all 16 features)
  d) SVM alone (all 16 features) — comparison
  e) LogReg alone (all 16 features) — comparison
  f) KDE-TS(elo) + SVM(all) + LogReg(elo) 3-way
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
from features import build_match_features, ELO_FEATURES, SQUAD_FEATURES, ALL_FEATURES, CONTEXT_FEATURES

ARTIFACTS_DIR = "/home/user/research/wave3-context/artifacts"
SEED = 0
N_SPLITS = 5
N_REPEATS = 10
BASELINE_LOSS = 0.8337
FRONTIER_LOSS = 0.7608
ELO_AND_SQUAD = ELO_FEATURES + SQUAD_FEATURES


class KDETwoStageModel:
    """KDE Stage 1 for draw probability (elo_diff only). LogReg Stage 2."""
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.2):
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()  # first feature = elo_diff
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


def cv_hetero_blend(X_elo, X_rich, y_enc, classes, kde_fn, svm_fn, alpha, exp_name, verbose=True):
    """Blend KDE-TS (trained on X_elo) with SVM (trained on X_rich)."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    n, nc = len(y_enc), len(classes)
    oof = np.zeros((n, nc))
    fl = []
    for ti, vi in cv.split(X_elo, y_enc):
        # KDE component (elo features)
        Xtr_elo, Xv_elo = X_elo[ti], X_elo[vi]
        ytr, yv = y_enc[ti], y_enc[vi]
        sc_elo = StandardScaler()
        Xtr_elo_s = sc_elo.fit_transform(Xtr_elo)
        Xv_elo_s  = sc_elo.transform(Xv_elo)
        kde = kde_fn(); kde.fit(Xtr_elo_s, ytr)
        p_kde = kde.predict_proba(Xv_elo_s)
        # SVM component (rich features)
        Xtr_rich, Xv_rich = X_rich[ti], X_rich[vi]
        sc_rich = StandardScaler()
        Xtr_rich_s = sc_rich.fit_transform(Xtr_rich)
        Xv_rich_s  = sc_rich.transform(Xv_rich)
        svm = svm_fn(); svm.fit(Xtr_rich_s, ytr)
        p_svm = svm.predict_proba(Xv_rich_s)
        bp = alpha * p_kde + (1 - alpha) * p_svm
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
           "verdict_vs_baseline": v, "label_classes": list(classes)}
    if verbose:
        print(f"\n{'='*60}\nExp: {exp_name}\nlog-loss: {ml:.4f} ± {sl:.4f}\nΔ base: {db:+.4f} (p={pb:.4f}) → {v}")
    od = os.path.join(ARTIFACTS_DIR, exp_name)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "metrics.json"), "w") as f:
        json.dump(met, f, indent=2)
    return met


def cv_single(X_feat, y_enc, classes, model_fn, exp_name, verbose=True):
    """CV with a single model (no blend)."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    n, nc = len(y_enc), len(classes)
    oof = np.zeros((n, nc))
    fl = []
    for ti, vi in cv.split(X_feat, y_enc):
        Xtr, Xv = X_feat[ti], X_feat[vi]
        ytr, yv = y_enc[ti], y_enc[vi]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xv_s  = sc.transform(Xv)
        m = model_fn(); m.fit(Xtr_s, ytr)
        bp = m.predict_proba(Xv_s)
        oof[vi] += bp / N_REPEATS
        fl.append(log_loss(yv, bp))
    ra = np.array(fl).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    ml, sl = ra.mean(), ra.std()
    pm = [-np.log(oof[i, y_enc[i]] + 1e-15) for i in range(n)]
    try: _, pb = wilcoxon(np.array(pm) - BASELINE_LOSS, alternative='less')
    except: pb = 1.0
    db = ml - BASELINE_LOSS
    v = "GREEN" if db < -0.01 and pb < 0.05 else ("RED" if db > 0.01 else "FLAT")
    if verbose:
        print(f"\n{'='*60}\nExp: {exp_name}\nlog-loss: {ml:.4f} ± {sl:.4f}\nΔ base: {db:+.4f} (p={pb:.4f}) → {v}")
    od = os.path.join(ARTIFACTS_DIR, exp_name)
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "metrics.json"), "w") as f:
        json.dump({"experiment": exp_name, "cv_log_loss_mean": round(ml, 4),
                   "cv_log_loss_std": round(sl, 4), "delta_vs_baseline_0.8337": round(db, 4),
                   "wilcoxon_vs_baseline_pvalue": round(pb, 4), "verdict_vs_baseline": v}, f, indent=2)
    return met if False else {"experiment": exp_name, "cv_log_loss_mean": round(ml, 4), "verdict_vs_baseline": v}


def run_experiments():
    print("Loading WC-2026 features...")
    df_elo  = build_match_features(include_context=False)
    df_full = build_match_features(include_context=True)
    y = df_elo["label"].values
    le = LabelEncoder()
    yenc = le.fit_transform(y)
    classes = le.classes_
    n = len(yenc)
    print(f"n={n}, classes={classes}")

    X_elo        = df_elo[ELO_FEATURES].values
    X_elo_squad  = df_elo[ELO_AND_SQUAD].values
    X_all        = df_full[[c for c in ALL_FEATURES if c in df_full.columns]].values

    kde_best = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)
    svm_std  = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)

    # ── W3C-19a: KDE-TS(elo) + SVM(all 16 features) ────────────────────────
    print(f"\n--- W3C-19a: KDE-TS(elo) + SVM(all {X_all.shape[1]} features) ---")
    best_19a = None
    for alpha in [0.4, 0.5, 0.6]:
        m = cv_hetero_blend(X_elo, X_all, yenc, classes, kde_best, svm_std,
                            alpha, f"W3C-19a_a{int(alpha*10)}", verbose=False)
        print(f"  alpha={alpha} (KDE weight): {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_19a is None or m['cv_log_loss_mean'] < best_19a['cv_log_loss_mean']:
            best_19a = m
    print(f"Best W3C-19a: {best_19a['cv_log_loss_mean']:.4f} → {best_19a['experiment']}")

    # ── W3C-19b: KDE-TS(elo) + SVM(squad+elo, 8 features) ──────────────────
    print(f"\n--- W3C-19b: KDE-TS(elo) + SVM(elo+squad {X_elo_squad.shape[1]} features) ---")
    best_19b = None
    for alpha in [0.4, 0.5, 0.6]:
        m = cv_hetero_blend(X_elo, X_elo_squad, yenc, classes, kde_best, svm_std,
                            alpha, f"W3C-19b_a{int(alpha*10)}", verbose=False)
        print(f"  alpha={alpha}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_19b is None or m['cv_log_loss_mean'] < best_19b['cv_log_loss_mean']:
            best_19b = m
    print(f"Best W3C-19b: {best_19b['cv_log_loss_mean']:.4f}")

    # ── W3C-19c: KDE-TS(elo) + LogReg(all features) ─────────────────────────
    print(f"\n--- W3C-19c: KDE-TS(elo) + LogReg(all {X_all.shape[1]} features) ---")
    lr_fn = lambda: LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    best_19c = None
    for alpha in [0.4, 0.5, 0.6]:
        m = cv_hetero_blend(X_elo, X_all, yenc, classes, kde_best, lr_fn,
                            alpha, f"W3C-19c_a{int(alpha*10)}", verbose=False)
        print(f"  alpha={alpha}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_19c is None or m['cv_log_loss_mean'] < best_19c['cv_log_loss_mean']:
            best_19c = m
    print(f"Best W3C-19c: {best_19c['cv_log_loss_mean']:.4f}")

    # ── W3C-19d: SVM alone (all features) ───────────────────────────────────
    print(f"\n--- W3C-19d: SVM alone (all {X_all.shape[1]} features) ---")
    m19d = cv_single(X_all, yenc, classes, svm_std, "W3C-19d_svm_all")

    # ── W3C-19e: LogReg alone (all features) ────────────────────────────────
    print(f"\n--- W3C-19e: LogReg alone (all {X_all.shape[1]} features) ---")
    m19e = cv_single(X_all, yenc, classes, lr_fn, "W3C-19e_lr_all")

    # ── W3C-19f: Best alpha blend + best SVM from 19d/19e ───────────────────
    print(f"\n--- W3C-19f: Fine alpha sweep with best config ---")
    # Use all features for SVM with best alpha from above
    best_19f = None
    for alpha in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        m = cv_hetero_blend(X_elo, X_all, yenc, classes, kde_best, svm_std,
                            alpha, f"W3C-19f_a{int(alpha*100)}", verbose=False)
        print(f"  alpha={alpha:.2f}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_19f is None or m['cv_log_loss_mean'] < best_19f['cv_log_loss_mean']:
            best_19f = m
    print(f"Best W3C-19f: {best_19f['cv_log_loss_mean']:.4f} → {best_19f['experiment']}")

    print("\nAll W3C-19 experiments done.")


if __name__ == "__main__":
    run_experiments()
