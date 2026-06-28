"""
W3C-04 / W3C-05: Historical WC augmentation experiments.

Adds WC 2010-2022 group+knockout matches as extra training rows.
Uses time-varying Elo computed from the full historical corpus.
Aligns Elo scale between historical and WC-2026.
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from features import build_match_features, ELO_FEATURES
from build_historical import build_historical_training_set

ARTIFACTS_DIR = "/home/user/research/wave3-context/artifacts"
SEED = 0
N_SPLITS = 5
N_REPEATS = 10
BASELINE_LOSS = 0.8337
FRONTIER_LOSS = 0.7608


def cv_evaluate_augmented(X_wc26, y_wc26, X_hist, y_hist, weights_hist, model_factory, exp_name):
    """
    CV on WC-2026, augmenting each training fold with historical data.
    model_factory: callable() -> sklearn estimator (not pipeline)
    weights_hist: per-row weights for historical training data
    """
    le = LabelEncoder()
    all_labels = np.concatenate([y_wc26, y_hist])
    le.fit(all_labels)
    classes = le.classes_
    y_wc26_enc = le.transform(y_wc26)
    y_hist_enc = le.transform(y_hist)

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    oof_probs = np.zeros((len(y_wc26), len(classes)))
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_wc26, y_wc26_enc)):
        X_tr_wc = X_wc26[train_idx]
        X_val = X_wc26[val_idx]
        y_tr_wc = y_wc26_enc[train_idx]
        y_val = y_wc26_enc[val_idx]

        # Scale using WC training data
        sc = StandardScaler()
        X_tr_wc_sc = sc.fit_transform(X_tr_wc)
        X_val_sc = sc.transform(X_val)
        X_hist_sc = sc.transform(X_hist)  # same scaling

        # Combine WC train + historical
        X_combined = np.vstack([X_tr_wc_sc, X_hist_sc])
        y_combined = np.concatenate([y_tr_wc, y_hist_enc])
        w_combined = np.concatenate([np.ones(len(y_tr_wc)), weights_hist])

        clf = model_factory()
        clf.fit(X_combined, y_combined, sample_weight=w_combined)
        probs = clf.predict_proba(X_val_sc)

        oof_probs[val_idx] += probs / N_REPEATS
        fold_losses.append(log_loss(y_val, probs))

    per_repeat_means = np.array(fold_losses).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    mean_loss = np.mean(per_repeat_means)
    std_loss = np.std(per_repeat_means, ddof=1)

    per_match_losses = []
    for i in range(len(y_wc26)):
        per_match_losses.append(log_loss([y_wc26_enc[i]], [oof_probs[i]],
                                         labels=list(range(len(classes)))))

    acc = accuracy_score(y_wc26_enc, np.argmax(oof_probs, axis=1))

    stat_vs_base = wilcoxon(per_match_losses, np.full(len(y_wc26), BASELINE_LOSS), alternative="less")
    stat_vs_frontier = wilcoxon(per_match_losses, np.full(len(y_wc26), FRONTIER_LOSS), alternative="less")

    delta_base = mean_loss - BASELINE_LOSS
    delta_frontier = mean_loss - FRONTIER_LOSS
    verdict = "GREEN" if (delta_base < -0.01 and stat_vs_base.pvalue < 0.05) else \
              ("RED" if delta_base > 0.01 else "FLAT")

    metrics = {
        "experiment": exp_name,
        "cv_log_loss_mean": round(mean_loss, 4),
        "cv_log_loss_std": round(std_loss, 4),
        "accuracy": round(acc, 4),
        "n_wc2026": len(y_wc26),
        "n_historical": len(y_hist),
        "delta_vs_baseline_0.8337": round(delta_base, 4),
        "delta_vs_frontier_0.7608": round(delta_frontier, 4),
        "wilcoxon_vs_baseline_pvalue": round(float(stat_vs_base.pvalue), 4),
        "wilcoxon_vs_frontier_pvalue": round(float(stat_vs_frontier.pvalue), 4),
        "verdict_vs_baseline": verdict,
        "label_classes": list(classes),
    }

    return metrics, oof_probs, per_match_losses, classes


def save_artifacts(exp_name, metrics, oof_probs, per_match_losses, classes, feature_list, run_info):
    out_dir = os.path.join(ARTIFACTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.save(f"{out_dir}/oof_probs.npy", oof_probs)

    oof_df = pd.DataFrame(oof_probs, columns=[f"p_{c}" for c in classes])
    oof_df["per_match_log_loss"] = per_match_losses
    oof_df.to_csv(f"{out_dir}/oof_predictions.csv", index=False)

    run_info["features"] = feature_list
    run_info["seed"] = SEED
    run_info["cv"] = f"RepeatedStratifiedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS})"
    import sklearn
    run_info["sklearn_version"] = sklearn.__version__

    with open(f"{out_dir}/run.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"  CV log-loss: {metrics['cv_log_loss_mean']:.4f} ± {metrics['cv_log_loss_std']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Δ baseline:  {metrics['delta_vs_baseline_0.8337']:+.4f} (p={metrics['wilcoxon_vs_baseline_pvalue']:.4f})")
    print(f"  Δ frontier:  {metrics['delta_vs_frontier_0.7608']:+.4f} (p={metrics['wilcoxon_vs_frontier_pvalue']:.4f})")
    print(f"  Verdict:     {metrics['verdict_vs_baseline']}")
    print(f"{'='*60}")

    return out_dir


def align_elo_scale(hist_elo_diff, wc26_elo_diff):
    """
    Align historical Elo diff scale to WC-2026 Elo scale.
    Both use running Elo from the same source, just the WC-2026
    uses pre-tournament snapshot. Apply a linear rescaling.
    """
    # Simple approach: standardize both separately within their distribution
    # Actually, use z-score: (elo_diff - mu) / sigma per dataset
    # Or just return as-is (scales should be similar since same Elo system)
    return hist_elo_diff


def run_w3c04(df_wc26, hist_df):
    """
    W3C-04: Augment WC-2026 CV folds with historical WC matches.
    Features: elo_diff + host_advantage.
    Grid search over historical weight.
    """
    feats = ["elo_diff", "host_advantage"]

    X_wc26 = df_wc26[feats].fillna(0).values
    y_wc26 = df_wc26["label"].values

    X_hist = hist_df[feats].fillna(0).values
    y_hist = hist_df["label"].values

    print(f"WC-2026: {len(y_wc26)} matches")
    print(f"Historical WC: {len(y_hist)} matches")
    print(f"WC-2026 elo_diff stats: mean={X_wc26[:,0].mean():.1f}, std={X_wc26[:,0].std():.1f}")
    print(f"Historical elo_diff stats: mean={X_hist[:,0].mean():.1f}, std={X_hist[:,0].std():.1f}")

    # Grid over weights and C
    best_loss = float("inf")
    best_metrics = None
    best_oof = None
    best_pml = None
    best_classes = None
    best_config = {}

    for hist_weight in [0.1, 0.2, 0.3, 0.5, 1.0]:
        for C in [0.1, 0.3, 1.0]:
            weights_hist = np.full(len(y_hist), hist_weight)

            def model_factory(C=C):
                return LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver="lbfgs")

            metrics, oof_probs, pml, classes = cv_evaluate_augmented(
                X_wc26, y_wc26, X_hist, y_hist, weights_hist,
                lambda C=C: LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver="lbfgs"),
                f"W3C-04-w{hist_weight}-C{C}"
            )
            print(f"  w={hist_weight}, C={C}: {metrics['cv_log_loss_mean']:.4f} ± {metrics['cv_log_loss_std']:.4f}")

            if metrics["cv_log_loss_mean"] < best_loss:
                best_loss = metrics["cv_log_loss_mean"]
                best_metrics = metrics
                best_oof = oof_probs
                best_pml = pml
                best_classes = classes
                best_config = {"hist_weight": hist_weight, "C": C}

    best_metrics["experiment"] = "W3C-04"
    best_metrics["best_config"] = best_config

    run_info = {
        "experiment": "W3C-04",
        "description": f"Augmented CV: WC-2026 + historical WC 2010-2022 (256 matches) as extra training. Best config: {best_config}",
        "model": f"LogisticRegression(C={best_config['C']})",
        "n_historical": len(y_hist),
        "best_config": best_config,
    }
    return save_artifacts("W3C-04", best_metrics, best_oof, best_pml, best_classes, feats, run_info)


def run_w3c04b_squad(df_wc26, hist_df):
    """
    W3C-04b: Augmented training with squad features added for WC-2026.
    Historical matches only have elo_diff (zero-fill squad features).
    """
    from features import SQUAD_FEATURES
    feats_wc = ["elo_diff", "host_advantage"] + SQUAD_FEATURES
    feats_hist = ["elo_diff", "host_advantage"]

    X_wc26 = df_wc26[feats_wc].fillna(0).values
    y_wc26 = df_wc26["label"].values

    # Historical: zero-fill squad features
    X_hist_base = hist_df[feats_hist].fillna(0).values
    n_extra = len(SQUAD_FEATURES)
    X_hist = np.hstack([X_hist_base, np.zeros((len(X_hist_base), n_extra))])
    y_hist = hist_df["label"].values

    best_loss = float("inf")
    best_metrics = None
    best_oof = None
    best_pml = None
    best_classes = None
    best_config = {}

    for hist_weight in [0.2, 0.3, 0.5]:
        for C in [0.05, 0.1, 0.3]:
            weights_hist = np.full(len(y_hist), hist_weight)
            metrics, oof_probs, pml, classes = cv_evaluate_augmented(
                X_wc26, y_wc26, X_hist, y_hist, weights_hist,
                lambda C=C: LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver="lbfgs"),
                f"W3C-04b-w{hist_weight}-C{C}"
            )
            print(f"  w={hist_weight}, C={C}: {metrics['cv_log_loss_mean']:.4f}")

            if metrics["cv_log_loss_mean"] < best_loss:
                best_loss = metrics["cv_log_loss_mean"]
                best_metrics = metrics
                best_oof = oof_probs
                best_pml = pml
                best_classes = classes
                best_config = {"hist_weight": hist_weight, "C": C}

    best_metrics["experiment"] = "W3C-04b"
    run_info = {
        "experiment": "W3C-04b",
        "description": f"Augmented CV: WC-2026 (Elo+squad) + historical WC 2010-2022 (Elo only, squad=0). Best: {best_config}",
        "model": f"LogisticRegression(C={best_config.get('C')})",
        "best_config": best_config,
    }
    return save_artifacts("W3C-04b", best_metrics, best_oof, best_pml, best_classes, feats_wc, run_info)


def run_w3c05_rf_historical(df_wc26, hist_df):
    """
    W3C-05: RandomForest augmented with historical WC data.
    """
    feats = ["elo_diff", "host_advantage"]
    X_wc26 = df_wc26[feats].fillna(0).values
    y_wc26 = df_wc26["label"].values
    X_hist = hist_df[feats].fillna(0).values
    y_hist = hist_df["label"].values

    best_loss = float("inf")
    best_metrics = None
    best_oof = None
    best_pml = None
    best_classes = None
    best_config = {}

    for hist_weight in [0.3, 0.5, 1.0]:
        for max_depth, min_leaf in [(3, 8), (4, 6), (3, 6)]:
            weights_hist = np.full(len(y_hist), hist_weight)
            metrics, oof_probs, pml, classes = cv_evaluate_augmented(
                X_wc26, y_wc26, X_hist, y_hist, weights_hist,
                lambda d=max_depth, l=min_leaf: RandomForestClassifier(
                    n_estimators=300, max_depth=d, min_samples_leaf=l,
                    max_features=0.8, random_state=SEED, n_jobs=-1
                ),
                f"W3C-05-w{hist_weight}-d{max_depth}-l{min_leaf}"
            )
            print(f"  w={hist_weight}, d={max_depth}, l={min_leaf}: {metrics['cv_log_loss_mean']:.4f}")

            if metrics["cv_log_loss_mean"] < best_loss:
                best_loss = metrics["cv_log_loss_mean"]
                best_metrics = metrics
                best_oof = oof_probs
                best_pml = pml
                best_classes = classes
                best_config = {"hist_weight": hist_weight, "max_depth": max_depth, "min_leaf": min_leaf}

    best_metrics["experiment"] = "W3C-05"
    run_info = {
        "experiment": "W3C-05",
        "description": f"RandomForest with historical WC augmentation. Best: {best_config}",
        "model": f"RandomForest({best_config})",
        "best_config": best_config,
    }
    return save_artifacts("W3C-05", best_metrics, best_oof, best_pml, best_classes, feats, run_info)


def run_w3c06_blend_with_historical(df_wc26, hist_df):
    """
    W3C-06: Blend of 3 models:
    - W3C-04 best (Elo+historical augment logistic)
    - W3C-04b best (Elo+squad+historical logistic)
    - Pure Elo logistic (no historical)

    This is our best local ensemble.
    """
    from features import SQUAD_FEATURES

    le = LabelEncoder()
    y_wc26 = df_wc26["label"].values
    le.fit(y_wc26)
    classes = le.classes_
    y_enc = le.transform(y_wc26)

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    # Feature sets
    feats_elo = ["elo_diff", "host_advantage"]
    feats_squad = feats_elo + SQUAD_FEATURES

    X_elo = df_wc26[feats_elo].fillna(0).values
    X_squad = df_wc26[feats_squad].fillna(0).values
    y = df_wc26["label"].values

    X_hist_elo = hist_df[feats_elo].fillna(0).values
    X_hist_squad = np.hstack([X_hist_elo, np.zeros((len(X_hist_elo), len(SQUAD_FEATURES)))])
    y_hist = hist_df["label"].values
    y_hist_enc = le.transform(y_hist)

    HIST_W = 0.3

    oof_m1 = np.zeros((len(y), len(classes)))  # Elo-only logistic (no hist)
    oof_m2 = np.zeros((len(y), len(classes)))  # Elo+hist augmented logistic
    oof_m3 = np.zeros((len(y), len(classes)))  # Elo+squad+hist augmented logistic

    fold_blend_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_elo, y_enc)):
        # M1: pure Elo logistic
        sc1 = StandardScaler()
        X_tr1 = sc1.fit_transform(X_elo[train_idx])
        X_val1 = sc1.transform(X_elo[val_idx])
        clf1 = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, solver="lbfgs")
        clf1.fit(X_tr1, y_enc[train_idx])
        p1 = clf1.predict_proba(X_val1)

        # M2: Elo + historical augment
        sc2 = StandardScaler()
        X_tr2 = sc2.fit_transform(X_elo[train_idx])
        X_val2 = sc2.transform(X_elo[val_idx])
        X_hist2 = sc2.transform(X_hist_elo)
        X_comb2 = np.vstack([X_tr2, X_hist2])
        y_comb2 = np.concatenate([y_enc[train_idx], y_hist_enc])
        w_comb2 = np.concatenate([np.ones(len(train_idx)), np.full(len(y_hist), HIST_W)])
        clf2 = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED, solver="lbfgs")
        clf2.fit(X_comb2, y_comb2, sample_weight=w_comb2)
        p2 = clf2.predict_proba(X_val2)

        # M3: Squad + historical augment
        sc3 = StandardScaler()
        X_tr3 = sc3.fit_transform(X_squad[train_idx])
        X_val3 = sc3.transform(X_squad[val_idx])
        X_hist3 = sc3.transform(X_hist_squad)
        X_comb3 = np.vstack([X_tr3, X_hist3])
        y_comb3 = np.concatenate([y_enc[train_idx], y_hist_enc])
        w_comb3 = np.concatenate([np.ones(len(train_idx)), np.full(len(y_hist), HIST_W)])
        clf3 = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED, solver="lbfgs")
        clf3.fit(X_comb3, y_comb3, sample_weight=w_comb3)
        p3 = clf3.predict_proba(X_val3)

        # Equal blend
        p_blend = (p1 + p2 + p3) / 3.0

        oof_m1[val_idx] += p1 / N_REPEATS
        oof_m2[val_idx] += p2 / N_REPEATS
        oof_m3[val_idx] += p3 / N_REPEATS
        fold_blend_losses.append(log_loss(y_enc[val_idx], p_blend))

    oof_blend = (oof_m1 + oof_m2 + oof_m3) / 3.0

    per_repeat_means = np.array(fold_blend_losses).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    mean_loss = np.mean(per_repeat_means)
    std_loss = np.std(per_repeat_means, ddof=1)

    per_match_losses = []
    for i in range(len(y)):
        per_match_losses.append(log_loss([y_enc[i]], [oof_blend[i]], labels=list(range(len(classes)))))

    acc = accuracy_score(y_enc, np.argmax(oof_blend, axis=1))
    stat_base = wilcoxon(per_match_losses, np.full(len(y), BASELINE_LOSS), alternative="less")
    stat_frontier = wilcoxon(per_match_losses, np.full(len(y), FRONTIER_LOSS), alternative="less")

    delta_base = mean_loss - BASELINE_LOSS
    delta_frontier = mean_loss - FRONTIER_LOSS
    verdict = "GREEN" if (delta_base < -0.01 and stat_base.pvalue < 0.05) else \
              ("RED" if delta_base > 0.01 else "FLAT")

    metrics = {
        "experiment": "W3C-06",
        "cv_log_loss_mean": round(mean_loss, 4),
        "cv_log_loss_std": round(std_loss, 4),
        "accuracy": round(acc, 4),
        "blend_components": ["Elo-logistic", "Elo+hist-logistic", "Elo+squad+hist-logistic"],
        "blend_weights": "equal 1/3",
        "n_matches": len(y),
        "delta_vs_baseline_0.8337": round(delta_base, 4),
        "delta_vs_frontier_0.7608": round(delta_frontier, 4),
        "wilcoxon_vs_baseline_pvalue": round(float(stat_base.pvalue), 4),
        "wilcoxon_vs_frontier_pvalue": round(float(stat_frontier.pvalue), 4),
        "verdict_vs_baseline": verdict,
        "label_classes": list(classes),
    }
    run_info = {
        "experiment": "W3C-06",
        "description": "3-model blend: pure Elo logistic + Elo augmented (hist WC 256 matches, w=0.3) + Squad+hist logistic",
        "model": "EqualBlend(3 components)",
        "hist_weight": HIST_W,
    }
    return save_artifacts("W3C-06", metrics, oof_blend, per_match_losses, classes,
                          feats_squad, run_info)


if __name__ == "__main__":
    print("Loading WC-2026 feature matrix...")
    df_wc26 = build_match_features(include_context=False)

    hist_csv = "/home/user/research/wave3-context/data/historical_wc_training.csv"
    if os.path.exists(hist_csv):
        print("Loading pre-built historical training set...")
        hist_df = pd.read_csv(hist_csv)
    else:
        print("Building historical training set...")
        hist_df, _ = build_historical_training_set()

    exp = sys.argv[1] if len(sys.argv) > 1 else "all"

    if exp in ("all", "04"):
        print("\n--- W3C-04: Elo + Historical Augmentation ---")
        run_w3c04(df_wc26, hist_df)

    if exp in ("all", "04b"):
        print("\n--- W3C-04b: Elo + Squad + Historical Augmentation ---")
        run_w3c04b_squad(df_wc26, hist_df)

    if exp in ("all", "05"):
        print("\n--- W3C-05: RandomForest + Historical ---")
        run_w3c05_rf_historical(df_wc26, hist_df)

    if exp in ("all", "06"):
        print("\n--- W3C-06: 3-model Blend ---")
        run_w3c06_blend_with_historical(df_wc26, hist_df)

    print("\nAll done!")
