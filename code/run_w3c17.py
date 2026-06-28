"""
W3C-17: New modeling directions beyond KDE hyperparameter tuning.
  a) Combined bw=450 + prior_weight=0.2
  b) 2D KDE for Stage 1 (elo_diff + host_advantage)
  c) KDE bandwidth ensemble (average across bw={200,300,450,600})
  d) Epanechnikov kernel sweep
  e) Adaptive-bw KDE (local density-based bandwidth)
  f) OOF stacking: LogReg meta on KDE-TS + SVM OOF probs
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
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.1, kernel="gaussian"):
        self.bw = bw
        self.C2 = C2
        self.prior_weight = prior_weight
        self.kernel = kernel

    def _weights(self, diff):
        if self.kernel == "epanechnikov":
            u = diff / self.bw
            return np.where(np.abs(u) <= 1.0, 0.75 * (1 - u**2), 0.0)
        return np.exp(-0.5 * (diff / self.bw) ** 2)

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()  # elo_diff
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
            w = self._weights(elo[j] - self.X_train)
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


class KDE2DTwoStageModel:
    """2D Gaussian KDE using elo_diff + host_advantage."""
    def __init__(self, bw_elo=450.0, bw_host=1.0, C2=2.0, prior_weight=0.2):
        self.bw_elo = bw_elo
        self.bw_host = bw_host
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, :2].copy()
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
        n = len(X)
        p_draw = np.zeros(n)
        for j in range(n):
            d_elo  = (X[j, 0] - self.X_train[:, 0]) / self.bw_elo
            d_host = (X[j, 1] - self.X_train[:, 1]) / self.bw_host
            w = np.exp(-0.5 * (d_elo**2 + d_host**2))
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


class AdaptiveBWKDEModel:
    """Adaptive-bandwidth KDE (pilot + local density scaling)."""
    def __init__(self, pilot_bw=450.0, C2=2.0, prior_weight=0.2, alpha=0.5):
        self.pilot_bw = pilot_bw
        self.C2 = C2
        self.prior_weight = prior_weight
        self.alpha = alpha

    def fit(self, X, y):
        self.X_train = X[:, 0].copy()
        self.y_draw = (y == 1).astype(float)
        self.prior_draw = self.y_draw.mean()
        n = len(self.X_train)
        pilot_dens = np.zeros(n)
        for i in range(n):
            w = np.exp(-0.5 * ((self.X_train[i] - self.X_train) / self.pilot_bw)**2)
            pilot_dens[i] = w.sum() / (n * self.pilot_bw)
        g = np.exp(np.log(np.clip(pilot_dens, 1e-30, None)).mean())
        self.lambdas = (pilot_dens / g) ** (-self.alpha)
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
            local_bw = self.pilot_bw * self.lambdas
            w = np.exp(-0.5 * ((elo[j] - self.X_train) / local_bw)**2)
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
    X = df[ELO_FEATURES].values  # ["elo_diff", "host_advantage"]
    y = df["label"].values
    le = LabelEncoder()
    yenc = le.fit_transform(y)
    classes = le.classes_
    print(f"n={len(yenc)}, classes={classes}")

    svm_fn = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)

    # ── W3C-17a: Combined bw=450 + prior_weight=0.2 ────────────────────────
    print("\n--- W3C-17a: KDE-TS(bw=450, pw=0.2) + SVM ---")
    m17a = cv_blend(X, yenc, classes,
        [("KDE_TS", lambda: KDETwoStageModel(450.0, 2.0, prior_weight=0.2)),
         ("SVM", svm_fn)],
        [0.5, 0.5], "W3C-17a_bw450_pw02")

    # ── W3C-17b: 2D KDE — sweep bw_host ────────────────────────────────────
    print("\n--- W3C-17b: 2D KDE Stage 1 (bw_host sweep, bw_elo=450) ---")
    best_17b = None
    for bw_host in [0.3, 0.5, 1.0, 2.0, 5.0]:
        m = cv_blend(X, yenc, classes,
            [("KDE2D", lambda bh=bw_host: KDE2DTwoStageModel(450.0, bh, 2.0, 0.2)),
             ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-17b_bwh{str(bw_host).replace('.','')}", verbose=False)
        print(f"  bw_host={bw_host}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_17b is None or m['cv_log_loss_mean'] < best_17b['cv_log_loss_mean']:
            best_17b = m
    print(f"Best W3C-17b: {best_17b['cv_log_loss_mean']:.4f} → {best_17b['experiment']}")

    # ── W3C-17c: Bandwidth ensemble (4 bw values averaged) ─────────────────
    print("\n--- W3C-17c: KDE bandwidth ensemble (bw={200,300,450,600}) ---")
    bw_vals = [200, 300, 450, 600]
    kde_w = 0.5 / len(bw_vals)
    comps_ens = [(f"KDE{bw}", lambda bw=bw: KDETwoStageModel(float(bw), 2.0, 0.2))
                 for bw in bw_vals] + [("SVM", svm_fn)]
    wts_ens = [kde_w] * len(bw_vals) + [0.5]
    m17c = cv_blend(X, yenc, classes, comps_ens, wts_ens, "W3C-17c_bw_ensemble")

    # ── W3C-17d: Epanechnikov kernel ────────────────────────────────────────
    print("\n--- W3C-17d: Epanechnikov kernel KDE-TS + SVM ---")
    best_17d = None
    for bw in [400, 500, 600, 700, 800]:
        m = cv_blend(X, yenc, classes,
            [("KDE_EP", lambda bw=bw: KDETwoStageModel(float(bw), 2.0, 0.2, kernel="epanechnikov")),
             ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-17d_ep_bw{bw}", verbose=False)
        print(f"  bw={bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_17d is None or m['cv_log_loss_mean'] < best_17d['cv_log_loss_mean']:
            best_17d = m
    print(f"Best W3C-17d: {best_17d['cv_log_loss_mean']:.4f}")

    # ── W3C-17e: Adaptive bandwidth ─────────────────────────────────────────
    print("\n--- W3C-17e: Adaptive-bandwidth KDE-TS + SVM ---")
    best_17e = None
    for alpha in [0.3, 0.5, 0.7]:
        for pilot_bw in [300, 450]:
            m = cv_blend(X, yenc, classes,
                [("AKD", lambda a=alpha, pb=pilot_bw: AdaptiveBWKDEModel(float(pb), 2.0, 0.2, a)),
                 ("SVM", svm_fn)],
                [0.5, 0.5], f"W3C-17e_a{str(alpha).replace('.','')}_pb{pilot_bw}", verbose=False)
            print(f"  alpha={alpha}, pilot_bw={pilot_bw}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
            if best_17e is None or m['cv_log_loss_mean'] < best_17e['cv_log_loss_mean']:
                best_17e = m
    print(f"Best W3C-17e: {best_17e['cv_log_loss_mean']:.4f}")

    # ── W3C-17f: OOF stacking ────────────────────────────────────────────────
    print("\n--- W3C-17f: OOF stacking (KDE-TS + SVM → LogReg meta) ---")
    cv_outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    nc = len(classes)
    oof_meta = np.zeros((len(yenc), nc))
    fl_meta = []
    for ti, vi in cv_outer.split(X, yenc):
        Xtr, Xv = X[ti], X[vi]
        ytr, yv = yenc[ti], yenc[vi]
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xv_s  = sc.transform(Xv)
        # Inner CV for OOF training predictions
        inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED+1)
        oof_tr = np.zeros((len(ti), 6))
        counts = np.zeros(len(ti))
        for i_ti, i_vi in inner_cv.split(Xtr_s, ytr):
            Xi_tr, Xi_v = Xtr_s[i_ti], Xtr_s[i_vi]
            yi_tr = ytr[i_ti]
            kde = KDETwoStageModel(450.0, 2.0, 0.2)
            svm = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
            kde.fit(Xi_tr, yi_tr); svm.fit(Xi_tr, yi_tr)
            oof_tr[i_vi, :3] += kde.predict_proba(Xi_v)
            oof_tr[i_vi, 3:] += svm.predict_proba(Xi_v)
            counts[i_vi] += 1
        oof_tr /= np.maximum(counts[:, None], 1)  # average across repeats
        # Renormalize so probabilities sum to 1
        oof_tr[:, :3] /= oof_tr[:, :3].sum(axis=1, keepdims=True)
        oof_tr[:, 3:] /= oof_tr[:, 3:].sum(axis=1, keepdims=True)
        # Meta-learner
        meta = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
        meta.fit(oof_tr, ytr)
        # Full train + test predictions
        kde_f = KDETwoStageModel(450.0, 2.0, 0.2)
        svm_f = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
        kde_f.fit(Xtr_s, ytr); svm_f.fit(Xtr_s, ytr)
        te_feats = np.hstack([kde_f.predict_proba(Xv_s), svm_f.predict_proba(Xv_s)])
        fp = meta.predict_proba(te_feats)
        oof_meta[vi] += fp / N_REPEATS
        fl_meta.append(log_loss(yv, fp))
    ra_m = np.array(fl_meta).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    ml_m, sl_m = ra_m.mean(), ra_m.std()
    pm_m = [-np.log(oof_meta[i, yenc[i]] + 1e-15) for i in range(len(yenc))]
    try: _, pb_m = wilcoxon(np.array(pm_m) - BASELINE_LOSS, alternative='less')
    except: pb_m = 1.0
    db_m = ml_m - BASELINE_LOSS
    v_m = "GREEN" if db_m < -0.01 and pb_m < 0.05 else ("RED" if db_m > 0.01 else "FLAT")
    print(f"\n{'='*60}\nExp: W3C-17f_oof_stack\nlog-loss: {ml_m:.4f} ± {sl_m:.4f}\nΔ base: {db_m:+.4f} (p={pb_m:.4f}) → {v_m}")
    od = os.path.join(ARTIFACTS_DIR, "W3C-17f_oof_stack")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "metrics.json"), "w") as f:
        json.dump({"experiment": "W3C-17f_oof_stack", "cv_log_loss_mean": round(ml_m, 4),
                   "cv_log_loss_std": round(sl_m, 4), "delta_vs_baseline_0.8337": round(db_m, 4),
                   "wilcoxon_vs_baseline_pvalue": round(pb_m, 4), "verdict_vs_baseline": v_m}, f, indent=2)

    print("\nAll W3C-17 experiments done.")


if __name__ == "__main__":
    run_experiments()
