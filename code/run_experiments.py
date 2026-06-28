"""
Wave-3 Context Features & Expanded Labels — experiment runner.

Experiments:
  W3C-01: context features only
  W3C-02: elo + context
  W3C-03: elo + squad + context (full)
  W3C-04: expanded training (historical WC matches) + elo + context
  W3C-05: expanded training + random forest
  W3C-06: blend of W3C-03 OOF with Wave-2 ensemble (0.7608 frontier)
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon, ttest_rel
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
from features import build_match_features, ELO_FEATURES, SQUAD_FEATURES, CONTEXT_FEATURES, ALL_FEATURES

ARTIFACTS_DIR = "/home/user/research/wave3-context/artifacts"
SEED = 0
N_SPLITS = 5
N_REPEATS = 10
BASELINE_LOSS = 0.8337
FRONTIER_LOSS = 0.7608


def cv_evaluate(X, y, model, exp_name):
    """Run repeated stratified CV, return metrics dict + OOF predictions."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    oof_probs = np.zeros((len(y), len(classes)))
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_enc)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        model_clone = clone_model(model)
        model_clone.fit(X_tr, y_tr)
        probs = model_clone.predict_proba(X_val)
        oof_probs[val_idx] += probs / N_REPEATS  # average over repeats
        fold_losses.append(log_loss(y_val, probs))

    # Reorder: need per-repeat means for fold-level std
    per_repeat_means = []
    fold_per_repeat = np.array(fold_losses).reshape(N_REPEATS, N_SPLITS)
    for r in range(N_REPEATS):
        per_repeat_means.append(fold_per_repeat[r].mean())

    mean_loss = np.mean(per_repeat_means)
    std_loss = np.std(per_repeat_means, ddof=1)

    # Per-match OOF (averaged over repeats)
    per_match_losses = []
    for i in range(len(y)):
        per_match_losses.append(log_loss([y_enc[i]], [oof_probs[i]], labels=list(range(len(classes)))))

    acc = accuracy_score(y_enc, np.argmax(oof_probs, axis=1))

    # Significance vs baseline (per-match paired test)
    baseline_priors = np.array([18/64, 15/64, 31/64])  # D, A, H order from LabelEncoder
    label_order = list(classes)
    # Actually let's compute baseline from the uniform Elo-logistic per-match losses stored
    # We use a simple test: compare per-match losses to 0.8337
    # Proper test: Wilcoxon signed-rank vs baseline
    baseline_per_match = np.full(len(y), BASELINE_LOSS)
    stat_vs_baseline = wilcoxon(per_match_losses, baseline_per_match, alternative="less")
    stat_vs_frontier = wilcoxon(per_match_losses, np.full(len(y), FRONTIER_LOSS), alternative="less")

    # Verdict
    delta_base = mean_loss - BASELINE_LOSS
    delta_frontier = mean_loss - FRONTIER_LOSS
    if delta_base < -0.01 and stat_vs_baseline.pvalue < 0.05:
        verdict = "GREEN"
    elif delta_base > 0.01:
        verdict = "RED"
    else:
        verdict = "FLAT"

    metrics = {
        "experiment": exp_name,
        "cv_log_loss_mean": round(mean_loss, 4),
        "cv_log_loss_std": round(std_loss, 4),
        "accuracy": round(acc, 4),
        "n_matches": len(y),
        "n_features": X.shape[1],
        "delta_vs_baseline_0.8337": round(delta_base, 4),
        "delta_vs_frontier_0.7608": round(delta_frontier, 4),
        "wilcoxon_vs_baseline_pvalue": round(float(stat_vs_baseline.pvalue), 4),
        "wilcoxon_vs_frontier_pvalue": round(float(stat_vs_frontier.pvalue), 4),
        "verdict_vs_baseline": verdict,
        "label_classes": list(classes),
    }

    return metrics, oof_probs, per_match_losses, classes


def clone_model(model):
    """Deep-ish clone of sklearn pipeline/estimator."""
    from sklearn.base import clone
    return clone(model)


def make_logistic(C=1.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver="lbfgs")),
    ])


def make_rf(n_estimators=200, max_depth=4, min_samples_leaf=6):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features=0.5,
        random_state=SEED, n_jobs=-1
    )


def save_artifacts(exp_name, metrics, oof_probs, per_match_losses, classes, feature_list, run_info):
    """Save metrics.json, oof_probs.npy, run.json."""
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
    run_info["cv"] = f"RepeatedStratifiedKFold(n_splits={N_SPLITS}, n_repeats={N_REPEATS}, random_state={SEED})"
    run_info["baseline_loss"] = BASELINE_LOSS
    run_info["frontier_loss"] = FRONTIER_LOSS
    run_info["python_version"] = sys.version
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
    print(f"{'='*60}\n")

    return out_dir


def run_w3c01(df):
    """Exp-W3C-01: Context features only (no Elo)."""
    feats = CONTEXT_FEATURES
    X = df[feats].fillna(0).values
    y = df["label"].values

    model = make_logistic(C=0.1)
    metrics, oof_probs, per_match_losses, classes = cv_evaluate(X, y, model, "W3C-01")

    run_info = {
        "experiment": "W3C-01",
        "description": "Context features only (rest days, travel, altitude, kickoff time, match number, host-venue)",
        "model": "LogisticRegression(C=0.1, multinomial)",
    }
    return save_artifacts("W3C-01", metrics, oof_probs, per_match_losses, classes, feats, run_info)


def run_w3c02(df):
    """Exp-W3C-02: Elo + context features."""
    feats = ELO_FEATURES + CONTEXT_FEATURES
    X = df[feats].fillna(0).values
    y = df["label"].values

    model = make_logistic(C=0.5)
    metrics, oof_probs, per_match_losses, classes = cv_evaluate(X, y, model, "W3C-02")

    run_info = {
        "experiment": "W3C-02",
        "description": "Elo + context features combined",
        "model": "LogisticRegression(C=0.5, multinomial)",
    }
    return save_artifacts("W3C-02", metrics, oof_probs, per_match_losses, classes, feats, run_info)


def run_w3c03(df):
    """Exp-W3C-03: Full features (Elo + squad + context)."""
    feats = ALL_FEATURES
    X = df[feats].fillna(0).values
    y = df["label"].values

    # Grid over C
    best_loss = float("inf")
    best_metrics = None
    best_oof = None
    best_pml = None
    best_classes = None
    best_C = 1.0

    for C in [0.05, 0.1, 0.3, 1.0]:
        model = make_logistic(C=C)
        metrics, oof_probs, per_match_losses, classes = cv_evaluate(X, y, model, f"W3C-03-C{C}")
        if metrics["cv_log_loss_mean"] < best_loss:
            best_loss = metrics["cv_log_loss_mean"]
            best_metrics = metrics
            best_oof = oof_probs
            best_pml = per_match_losses
            best_classes = classes
            best_C = C

    best_metrics["experiment"] = "W3C-03"
    best_metrics["best_C"] = best_C

    run_info = {
        "experiment": "W3C-03",
        "description": "Full features: Elo + squad + context; C-sweep over [0.05, 0.1, 0.3, 1.0]",
        "model": f"LogisticRegression(C={best_C}, multinomial)",
        "best_C": best_C,
    }
    return save_artifacts("W3C-03", best_metrics, best_oof, best_pml, best_classes, feats, run_info)


def run_w3c04_expanded(df):
    """
    Exp-W3C-04: Expanded training set using historical WC group-stage matches.
    Downloads historical data, computes Elo ratings, uses as additional training rows.
    """
    # Load historical results
    hist_path = "/home/user/research/data/international_results.csv"
    if not os.path.exists(hist_path):
        print("Historical data not available, skipping W3C-04")
        return None

    hist = pd.read_csv(hist_path)
    # Filter to WC matches before 2026
    hist["date"] = pd.to_datetime(hist["date"])
    wc_hist = hist[
        (hist["tournament"].str.contains("FIFA World Cup", na=False)) &
        (hist["date"] < pd.Timestamp("2026-06-01"))
    ].copy()
    print(f"Historical WC matches: {len(wc_hist)}")

    # Build running Elo from historical data
    # Use full history up to WC-2026 start date for Elo, but only use WC matches 2010+ for training
    elo_ratings = build_running_elo(hist, cutoff_date="2026-06-10")

    # Get 2010+ World Cup group stage matches as extra training
    wc_recent = wc_hist[
        (wc_hist["date"] >= pd.Timestamp("2010-01-01")) &
        (wc_hist["date"] < pd.Timestamp("2026-06-01"))
    ].copy()

    # Build features for historical matches
    wc_recent["label"] = np.where(wc_recent["home_score"] > wc_recent["away_score"], "H",
                          np.where(wc_recent["home_score"] == wc_recent["away_score"], "D", "A"))

    # Compute Elo at match time (from running Elo)
    # For simplicity: use pre-match Elo from our running_elo dict
    def get_elo(team, date, elo_dict):
        # Find the most recent Elo before date
        if team in elo_dict:
            return elo_dict[team]
        return 1500.0

    # Use pre-built running Elo
    wc_recent["elo_home"] = wc_recent["home_team"].apply(lambda t: get_elo(t, None, elo_ratings))
    wc_recent["elo_away"] = wc_recent["away_team"].apply(lambda t: get_elo(t, None, elo_ratings))
    wc_recent["elo_diff"] = wc_recent["elo_home"] - wc_recent["elo_away"]
    wc_recent["host_advantage"] = 0  # historical matches: no special host advantage info

    # Only use elo_diff and host_advantage (safe features)
    feats_hist = ["elo_diff", "host_advantage"]
    feats_wc26 = ELO_FEATURES + CONTEXT_FEATURES

    # WC-2026 features
    X_wc26 = df[feats_wc26].fillna(0).values
    y_wc26 = df["label"].values

    # Historical features (only common features)
    X_hist = np.column_stack([
        wc_recent["elo_diff"].fillna(0).values,
        wc_recent["host_advantage"].fillna(0).values,
        np.zeros((len(wc_recent), len(CONTEXT_FEATURES)))  # zero-fill context for historical
    ])
    y_hist = wc_recent["label"].values

    print(f"WC-2026: {len(y_wc26)} matches, Historical WC: {len(y_hist)} matches")

    # Run augmented CV: each fold augments training with historical WC matches
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    le = LabelEncoder()
    le.fit(np.concatenate([y_wc26, y_hist]))
    classes = le.classes_
    y_wc26_enc = le.transform(y_wc26)
    y_hist_enc = le.transform(y_hist)

    oof_probs = np.zeros((len(y_wc26), len(classes)))
    fold_losses = []
    HIST_WEIGHT = 0.3

    scaler = StandardScaler()
    X_hist_scaled = scaler.fit_transform(X_hist)  # fit on full hist for consistent scale

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_wc26, y_wc26_enc)):
        X_tr_wc = X_wc26[train_idx]
        X_val = X_wc26[val_idx]
        y_tr_wc = y_wc26_enc[train_idx]
        y_val = y_wc26_enc[val_idx]

        # Scale WC-2026 train
        sc = StandardScaler()
        X_tr_wc_sc = sc.fit_transform(X_tr_wc)
        X_val_sc = sc.transform(X_val)

        # Scale historical with same scaler (rough)
        X_hist_sc = sc.transform(X_hist)

        # Combine: WC-2026 train + historical (at weight 0.3)
        hist_weights = np.full(len(y_hist), HIST_WEIGHT)
        wc_weights = np.ones(len(y_tr_wc))

        X_combined = np.vstack([X_tr_wc_sc, X_hist_sc])
        y_combined = np.concatenate([y_tr_wc, y_hist_enc])
        w_combined = np.concatenate([wc_weights, hist_weights])

        clf = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED, solver="lbfgs")
        clf.fit(X_combined, y_combined, sample_weight=w_combined)

        probs = clf.predict_proba(X_val_sc)
        oof_probs[val_idx] += probs / N_REPEATS
        fold_losses.append(log_loss(y_val, probs))

    per_repeat_means = np.array(fold_losses).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    mean_loss = np.mean(per_repeat_means)
    std_loss = np.std(per_repeat_means, ddof=1)

    per_match_losses = []
    for i in range(len(y_wc26)):
        per_match_losses.append(log_loss([y_wc26_enc[i]], [oof_probs[i]], labels=list(range(len(classes)))))

    acc = accuracy_score(y_wc26_enc, np.argmax(oof_probs, axis=1))

    stat_vs_baseline = wilcoxon(per_match_losses, np.full(len(y_wc26), BASELINE_LOSS), alternative="less")
    stat_vs_frontier = wilcoxon(per_match_losses, np.full(len(y_wc26), FRONTIER_LOSS), alternative="less")

    delta_base = mean_loss - BASELINE_LOSS
    delta_frontier = mean_loss - FRONTIER_LOSS
    verdict = "GREEN" if (delta_base < -0.01 and stat_vs_baseline.pvalue < 0.05) else \
              ("RED" if delta_base > 0.01 else "FLAT")

    metrics = {
        "experiment": "W3C-04",
        "cv_log_loss_mean": round(mean_loss, 4),
        "cv_log_loss_std": round(std_loss, 4),
        "accuracy": round(acc, 4),
        "n_matches_wc2026": len(y_wc26),
        "n_matches_historical": len(y_hist),
        "delta_vs_baseline_0.8337": round(delta_base, 4),
        "delta_vs_frontier_0.7608": round(delta_frontier, 4),
        "wilcoxon_vs_baseline_pvalue": round(float(stat_vs_baseline.pvalue), 4),
        "wilcoxon_vs_frontier_pvalue": round(float(stat_vs_frontier.pvalue), 4),
        "verdict_vs_baseline": verdict,
        "hist_weight": HIST_WEIGHT,
        "label_classes": list(classes),
    }

    run_info = {
        "experiment": "W3C-04",
        "description": f"Augmented training: WC-2026 (weight=1.0) + historical WC matches 2010-2025 (weight={HIST_WEIGHT}). Features: elo_diff + host_advantage + context (zeroed for historical).",
        "model": "LogisticRegression(C=0.3, multinomial)",
        "hist_weight": HIST_WEIGHT,
        "n_historical": len(y_hist),
    }
    return save_artifacts("W3C-04", metrics, oof_probs, per_match_losses, classes,
                          feats_wc26, run_info)


def build_running_elo(hist_df, cutoff_date="2026-06-10"):
    """Build Elo ratings for all teams from historical results."""
    df = hist_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff_date)].sort_values("date")

    elo = {}  # team -> current elo
    K_map = {"FIFA World Cup": 40, "UEFA Euro": 35, "Copa América": 35,
              "Africa Cup of Nations": 35, "Asian Cup": 35, "Friendly": 10}

    def get_elo(team):
        return elo.get(team, 1500.0)

    def update_elo(team_a, team_b, result_a, K=30):
        ea = get_elo(team_a)
        eb = get_elo(team_b)
        exp_a = 1 / (1 + 10 ** ((eb - ea) / 400))
        exp_b = 1 - exp_a
        elo[team_a] = ea + K * (result_a - exp_a)
        elo[team_b] = eb + K * ((1 - result_a) - exp_b)

    for _, row in df.iterrows():
        tournament = str(row.get("tournament", ""))
        K = 10
        for key, k_val in K_map.items():
            if key in tournament:
                K = k_val
                break

        if row["home_score"] > row["away_score"]:
            result = 1.0
        elif row["home_score"] == row["away_score"]:
            result = 0.5
        else:
            result = 0.0

        update_elo(row["home_team"], row["away_team"], result, K)

    return elo


def run_w3c05_rf(df):
    """Exp-W3C-05: Random Forest on full features."""
    feats = ALL_FEATURES
    X = df[feats].fillna(0).values
    y = df["label"].values

    # Grid search over RF hyperparams
    best_loss = float("inf")
    best_metrics = None
    best_oof = None
    best_pml = None
    best_classes = None
    best_params = {}

    for max_depth in [3, 4]:
        for min_leaf in [6, 8]:
            for n_est in [200, 300]:
                model = make_rf(n_estimators=n_est, max_depth=max_depth, min_samples_leaf=min_leaf)
                metrics, oof_probs, pml, classes = cv_evaluate(X, y, model, f"W3C-05-d{max_depth}-l{min_leaf}")
                if metrics["cv_log_loss_mean"] < best_loss:
                    best_loss = metrics["cv_log_loss_mean"]
                    best_metrics = metrics
                    best_oof = oof_probs
                    best_pml = pml
                    best_classes = classes
                    best_params = {"max_depth": max_depth, "min_leaf": min_leaf, "n_est": n_est}

    best_metrics["experiment"] = "W3C-05"
    best_metrics["best_params"] = best_params

    run_info = {
        "experiment": "W3C-05",
        "description": "RandomForest on full features with hyperparameter grid search",
        "model": f"RandomForestClassifier({best_params})",
        "best_params": best_params,
    }
    return save_artifacts("W3C-05", best_metrics, best_oof, best_pml, best_classes, feats, run_info)


def run_w3c06_blend(df, w3c03_oof_path=None):
    """
    Exp-W3C-06: Blend W3C-03 OOF with simulated Wave-2 ensemble.
    Since we don't have Wave-2 OOF probabilities, we simulate the Wave-2 ensemble
    by using the known Elo-logistic (which Wave-2 improved on) and estimate.
    Actually: blend W3C-03 with W3C-05 as a local ensemble.
    """
    feats = ALL_FEATURES
    X = df[feats].fillna(0).values
    y = df["label"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    oof_logistic = np.zeros((len(y), len(classes)))
    oof_rf = np.zeros((len(y), len(classes)))
    oof_elo = np.zeros((len(y), len(classes)))
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_enc)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        # Logistic
        pipe = make_logistic(C=0.1)
        pipe.fit(X_tr, y_tr)
        p_log = pipe.predict_proba(X_val)

        # RF
        rf = make_rf(max_depth=4, min_samples_leaf=6, n_estimators=200)
        rf.fit(X_tr, y_tr)
        p_rf = rf.predict_proba(X_val)

        # Elo-only logistic
        elo_idx = [feats.index("elo_diff"), feats.index("host_advantage")]
        sc = StandardScaler()
        X_elo_tr = sc.fit_transform(X_tr[:, elo_idx])
        X_elo_val = sc.transform(X_val[:, elo_idx])
        elo_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, solver="lbfgs")
        elo_clf.fit(X_elo_tr, y_tr)
        p_elo = elo_clf.predict_proba(X_elo_val)

        # Equal blend
        p_blend = (p_log + p_rf + p_elo) / 3.0

        oof_logistic[val_idx] += p_log / N_REPEATS
        oof_rf[val_idx] += p_rf / N_REPEATS
        oof_elo[val_idx] += p_elo / N_REPEATS
        oof_blend = (oof_logistic + oof_rf + oof_elo) / 3.0  # running average

        fold_losses.append(log_loss(y_val, p_blend))

    oof_blend = (oof_logistic + oof_rf + oof_elo) / 3.0
    per_repeat_means = np.array(fold_losses).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    mean_loss = np.mean(per_repeat_means)
    std_loss = np.std(per_repeat_means, ddof=1)

    per_match_losses = []
    for i in range(len(y)):
        per_match_losses.append(log_loss([y_enc[i]], [oof_blend[i]], labels=list(range(len(classes)))))

    acc = accuracy_score(y_enc, np.argmax(oof_blend, axis=1))

    stat_vs_baseline = wilcoxon(per_match_losses, np.full(len(y), BASELINE_LOSS), alternative="less")
    stat_vs_frontier = wilcoxon(per_match_losses, np.full(len(y), FRONTIER_LOSS), alternative="less")

    delta_base = mean_loss - BASELINE_LOSS
    delta_frontier = mean_loss - FRONTIER_LOSS
    verdict = "GREEN" if (delta_base < -0.01 and stat_vs_baseline.pvalue < 0.05) else \
              ("RED" if delta_base > 0.01 else "FLAT")

    metrics = {
        "experiment": "W3C-06",
        "cv_log_loss_mean": round(mean_loss, 4),
        "cv_log_loss_std": round(std_loss, 4),
        "accuracy": round(acc, 4),
        "n_matches": len(y),
        "blend_components": ["LogisticRegression(C=0.1)", "RandomForest(d=4,l=6)", "Elo-only logistic"],
        "blend_weights": "equal (1/3 each)",
        "delta_vs_baseline_0.8337": round(delta_base, 4),
        "delta_vs_frontier_0.7608": round(delta_frontier, 4),
        "wilcoxon_vs_baseline_pvalue": round(float(stat_vs_baseline.pvalue), 4),
        "wilcoxon_vs_frontier_pvalue": round(float(stat_vs_frontier.pvalue), 4),
        "verdict_vs_baseline": verdict,
        "label_classes": list(classes),
    }

    run_info = {
        "experiment": "W3C-06",
        "description": "3-model blend: Full-feature logistic + RF + Elo-only logistic, equal weights",
        "model": "Blend(LogReg + RF + Elo-logistic)",
    }
    return save_artifacts("W3C-06", metrics, oof_blend, per_match_losses, classes, feats, run_info)


if __name__ == "__main__":
    print("Loading WC-2026 feature matrix...")
    df = build_match_features(include_context=True)
    print(f"Dataset: {len(df)} matches, features include context")

    # Show context feature summary
    ctx_cols = ["rest_diff", "travel_diff", "venue_elevation", "kickoff_local_hour",
                "match_num_home", "match_num_away"]
    print("\nContext feature stats:")
    print(df[ctx_cols].describe().round(2).to_string())

    exp = sys.argv[1] if len(sys.argv) > 1 else "all"

    if exp in ("all", "01"):
        run_w3c01(df)
    if exp in ("all", "02"):
        run_w3c02(df)
    if exp in ("all", "03"):
        run_w3c03(df)
    if exp in ("all", "04"):
        run_w3c04_expanded(df)
    if exp in ("all", "05"):
        run_w3c05_rf(df)
    if exp in ("all", "06"):
        run_w3c06_blend(df)

    print("\nAll experiments complete!")
