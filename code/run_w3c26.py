"""
W3C-26: Confirm frontier with fixed SVC random_state + Student-t 2D KDE.
  a) Reconfirm Gaussian(bw=300)+SVM(rs=0) with fine alpha sweep [0.4,0.5,0.6]
  b) Reconfirm Student-t(nu=10, bw=200)+SVM(rs=0) with fine alpha sweep
  c) 2D Student-t KDE (elo_diff + host_advantage) Stage 1 + SVM
  d) Student-t(nu=15, bw=225) + SVM alpha sweep (promising from W3C-25c)
  e) Kernel Mixture: Gaussian(bw=300) + Student-t(nu=10,bw=200) stage 1 blend + SVM
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
    def __init__(self, bw=300.0, C2=2.0, prior_weight=0.2):
        self.bw = bw
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


class StudentTKDETwoStageModel:
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


class StudentT2DKDETwoStageModel:
    """2D Student-t KDE: uses both elo_diff AND host_advantage in Stage 1 kernel."""
    def __init__(self, bw_elo=200.0, bw_host=1.0, nu=10, C2=2.0, prior_weight=0.2):
        self.bw_elo = bw_elo
        self.bw_host = bw_host
        self.nu = nu
        self.C2 = C2
        self.prior_weight = prior_weight

    def fit(self, X, y):
        self.X_train = X[:, :2].copy()  # elo_diff, host_advantage
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
        elo = X[:, 0]
        host = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
        n = len(elo)
        p_draw = np.zeros(n)
        for j in range(n):
            d_elo = (elo[j] - self.X_train[:, 0]) / self.bw_elo
            d_host = (host[j] - self.X_train[:, 1]) / self.bw_host
            t2 = d_elo ** 2 + d_host ** 2
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


class KernelMixtureKDETwoStageModel:
    """Stage 1: weighted blend of Gaussian and Student-t kernel estimates."""
    def __init__(self, bw_g=300.0, bw_st=200.0, nu_st=10, alpha=0.5, C2=2.0, prior_weight=0.2):
        self.bw_g = bw_g
        self.bw_st = bw_st
        self.nu_st = nu_st
        self.alpha = alpha  # weight on Gaussian
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
            d = elo[j] - self.X_train
            wg = np.exp(-0.5 * (d / self.bw_g) ** 2)
            t2 = (d / self.bw_st) ** 2
            wst = (1.0 + t2 / self.nu_st) ** (-(self.nu_st + 1) / 2.0)
            w = self.alpha * wg + (1 - self.alpha) * wst
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


def cv_blend(X, y_enc, classes, components, blend_weights, exp_name, verbose=True, svm_seed=None):
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

    # Fixed random_state SVM for reproducibility
    svm_fn = lambda: SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED)

    # ── W3C-26a: Gaussian + SVM(rs=0) alpha sweep (reconfirm frontier) ────────
    print("\n--- W3C-26a: Gaussian(bw=300)+SVM(rs=0) alpha sweep ---")
    best_26a = None
    for alpha in [0.4, 0.45, 0.5, 0.55, 0.6]:
        kde_fn = lambda: KDETwoStageModel(bw=300.0, C2=2.0, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("KDE_TS", kde_fn), ("SVM", svm_fn)],
            [alpha, 1 - alpha], f"W3C-26a_gauss_a{int(alpha*100)}")
        print(f"  alpha={alpha:.2f}: {m['cv_log_loss_mean']:.4f} ± {m['cv_log_loss_std']:.4f} → {m['verdict_vs_baseline']}")
        if best_26a is None or m['cv_log_loss_mean'] < best_26a['cv_log_loss_mean']:
            best_26a = m
    print(f"Best W3C-26a (Gaussian): {best_26a['cv_log_loss_mean']:.4f}")

    # ── W3C-26b: Student-t(nu=10,bw=200)+SVM(rs=0) alpha sweep ──────────────
    print("\n--- W3C-26b: Student-t(nu=10,bw=200)+SVM(rs=0) alpha sweep ---")
    best_26b = None
    for alpha in [0.5, 0.55, 0.6, 0.65, 0.7]:
        st_fn = lambda: StudentTKDETwoStageModel(bw=200, nu=10, C2=2.0, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn), ("SVM", svm_fn)],
            [alpha, 1 - alpha], f"W3C-26b_st_a{int(alpha*100)}", verbose=False)
        print(f"  alpha={alpha:.2f}: {m['cv_log_loss_mean']:.4f} ± {m['cv_log_loss_std']:.4f} → {m['verdict_vs_baseline']}")
        if best_26b is None or m['cv_log_loss_mean'] < best_26b['cv_log_loss_mean']:
            best_26b = m
    print(f"Best W3C-26b (Student-t): {best_26b['cv_log_loss_mean']:.4f}")

    # ── W3C-26c: 2D Student-t KDE (elo+host) + SVM ───────────────────────────
    print("\n--- W3C-26c: 2D Student-t KDE + SVM ---")
    best_26c = None
    for bw_h in [0.3, 0.5, 1.0, 2.0]:
        st2d_fn = lambda bh=bw_h: StudentT2DKDETwoStageModel(bw_elo=200, bw_host=bh, nu=10, C2=2.0, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("St2DKDE", st2d_fn), ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-26c_2dst_bh{str(bw_h).replace('.','')}", verbose=False)
        print(f"  bw_host={bw_h}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_26c is None or m['cv_log_loss_mean'] < best_26c['cv_log_loss_mean']:
            best_26c = m
    print(f"Best W3C-26c: {best_26c['cv_log_loss_mean']:.4f}")

    # ── W3C-26d: Student-t(nu=15, bw=225) + SVM alpha sweep ─────────────────
    print("\n--- W3C-26d: Student-t(nu=15,bw=225)+SVM alpha sweep ---")
    best_26d = None
    for alpha in [0.4, 0.5, 0.6, 0.7]:
        st_fn = lambda: StudentTKDETwoStageModel(bw=225, nu=15, C2=2.0, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("StKDE", st_fn), ("SVM", svm_fn)],
            [alpha, 1 - alpha], f"W3C-26d_nu15_bw225_a{int(alpha*10)}", verbose=False)
        print(f"  alpha={alpha:.2f}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_26d is None or m['cv_log_loss_mean'] < best_26d['cv_log_loss_mean']:
            best_26d = m
    print(f"Best W3C-26d: {best_26d['cv_log_loss_mean']:.4f}")

    # ── W3C-26e: Kernel mixture (Gaussian+Student-t) + SVM ───────────────────
    print("\n--- W3C-26e: Gaussian+Student-t kernel mixture + SVM ---")
    best_26e = None
    for alpha_g in [0.3, 0.5, 0.7]:
        km_fn = lambda ag=alpha_g: KernelMixtureKDETwoStageModel(
            bw_g=300, bw_st=200, nu_st=10, alpha=ag, C2=2.0, prior_weight=0.2)
        m = cv_blend(X, yenc, classes,
            [("KM_KDE", km_fn), ("SVM", svm_fn)],
            [0.5, 0.5], f"W3C-26e_km_ag{int(alpha_g*10)}", verbose=False)
        print(f"  alpha_g={alpha_g}: {m['cv_log_loss_mean']:.4f} → {m['verdict_vs_baseline']}")
        if best_26e is None or m['cv_log_loss_mean'] < best_26e['cv_log_loss_mean']:
            best_26e = m
    print(f"Best W3C-26e: {best_26e['cv_log_loss_mean']:.4f}")

    print("\nAll W3C-26 experiments done.")
    all_bests = [best_26a, best_26b, best_26c, best_26d, best_26e]
    overall = min(all_bests, key=lambda x: x['cv_log_loss_mean'])
    print(f"\nOverall W3C-26 best: {overall['cv_log_loss_mean']:.4f} → {overall['experiment']}")


if __name__ == "__main__":
    run_experiments()
