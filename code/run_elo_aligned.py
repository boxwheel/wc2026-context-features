"""
W3C-04c: Historical augmentation with Elo SCALE ALIGNMENT.

Aligns historical running Elo to WC-2026 teams.csv Elo scale
using linear regression on the WC-2026 teams as anchor points.
Then uses these aligned historical Elo diffs as augmented training data.

Also tries WC-2022-only augmentation (most recent, most similar format).
"""
import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from features import build_match_features, ELO_FEATURES, SQUAD_FEATURES
from build_historical import build_historical_training_set, build_running_elo

ARTIFACTS_DIR = "/home/user/research/wave3-context/artifacts"
SEED = 0
N_SPLITS = 5
N_REPEATS = 10
BASELINE_LOSS = 0.8337
FRONTIER_LOSS = 0.7608


def compute_elo_alignment(df_wc26, final_elo, teams_df):
    """
    Compute linear mapping: my_running_elo -> teams.csv Elo
    using WC-2026 teams as anchor.
    """
    # Get teams and their provided Elo
    team_name_to_code = teams_df.set_index("team_name")["fifa_code"].to_dict()

    # Team names in historical data don't always match WC-2026 team names
    # Let's map common names
    name_map = {
        "United States": "USA", "South Korea": "KOR", "Czech Republic": "CZE",
        "Republic of Ireland": "IRL", "Iran": "IRN", "Serbia": "SRB",
        "Morocco": "MAR", "Senegal": "SEN", "Australia": "AUS",
        "Saudi Arabia": "KSA", "Türkiye": "TUR", "Turkey": "TUR",
        "DR Congo": "COD", "Cameroon": "CMR", "Nigeria": "NGA",
        "Ghana": "GHA", "Ivory Coast": "CIV", "Bosnia and Herzegovina": "BIH",
    }

    teams_df["team_name_norm"] = teams_df["team_name"].apply(
        lambda x: name_map.get(x, x))

    my_elo_list = []
    provided_elo_list = []

    for _, row in teams_df.iterrows():
        team_name = row["team_name"]
        norm_name = name_map.get(team_name, team_name)
        my_e = final_elo.get(team_name, final_elo.get(norm_name, None))
        if my_e is not None:
            my_elo_list.append(my_e)
            provided_elo_list.append(row["elo_rating"])

    my_elo_arr = np.array(my_elo_list).reshape(-1, 1)
    prov_elo_arr = np.array(provided_elo_list)

    lr = LinearRegression()
    lr.fit(my_elo_arr, prov_elo_arr)
    print(f"Elo alignment: provided_elo = {lr.coef_[0]:.4f} * my_elo + {lr.intercept_:.1f}")
    print(f"R² = {lr.score(my_elo_arr, prov_elo_arr):.4f}")
    print(f"Mapped {len(my_elo_list)} teams")

    return lr


def apply_elo_alignment(hist_df, lr, results_df=None):
    """Apply linear alignment to historical Elo diffs."""
    # The historical elo_diff = home_elo - away_elo (in my_elo scale)
    # Aligned: aligned_elo_h = lr.coef_[0] * my_elo_h + lr.intercept_
    # Aligned diff = coef_ * (my_elo_h - my_elo_a) = coef_ * my_elo_diff
    coef = lr.coef_[0]
    hist_df = hist_df.copy()
    hist_df["elo_diff"] = coef * hist_df["elo_diff"]
    return hist_df


def cv_evaluate_augmented(X_wc26, y_wc26, X_hist, y_hist, hist_weight,
                           model_factory, exp_name):
    le = LabelEncoder()
    le.fit(np.concatenate([y_wc26, y_hist]))
    classes = le.classes_
    y_enc = le.transform(y_wc26)
    y_hist_enc = le.transform(y_hist)

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    oof_probs = np.zeros((len(y_wc26), len(classes)))
    fold_losses = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_wc26, y_enc)):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_wc26[train_idx])
        X_val = sc.transform(X_wc26[val_idx])
        X_hist_sc = sc.transform(X_hist)

        X_comb = np.vstack([X_tr, X_hist_sc])
        y_comb = np.concatenate([y_enc[train_idx], y_hist_enc])
        w_comb = np.concatenate([np.ones(len(train_idx)), np.full(len(y_hist), hist_weight)])

        clf = model_factory()
        clf.fit(X_comb, y_comb, sample_weight=w_comb)
        p = clf.predict_proba(X_val)
        oof_probs[val_idx] += p / N_REPEATS
        fold_losses.append(log_loss(y_enc[val_idx], p))

    per_repeat_means = np.array(fold_losses).reshape(N_REPEATS, N_SPLITS).mean(axis=1)
    mean_loss = np.mean(per_repeat_means)
    std_loss = np.std(per_repeat_means, ddof=1)

    per_match_losses = [log_loss([y_enc[i]], [oof_probs[i]], labels=list(range(len(classes))))
                        for i in range(len(y_wc26))]
    acc = accuracy_score(y_enc, np.argmax(oof_probs, axis=1))

    stat_base = wilcoxon(per_match_losses, np.full(len(y_wc26), BASELINE_LOSS), alternative="less")
    stat_front = wilcoxon(per_match_losses, np.full(len(y_wc26), FRONTIER_LOSS), alternative="less")

    delta_base = mean_loss - BASELINE_LOSS
    delta_front = mean_loss - FRONTIER_LOSS
    verdict = "GREEN" if (delta_base < -0.01 and stat_base.pvalue < 0.05) else \
              ("RED" if delta_base > 0.01 else "FLAT")

    return {
        "cv_log_loss_mean": round(mean_loss, 4),
        "cv_log_loss_std": round(std_loss, 4),
        "accuracy": round(acc, 4),
        "delta_vs_baseline_0.8337": round(delta_base, 4),
        "delta_vs_frontier_0.7608": round(delta_front, 4),
        "wilcoxon_vs_baseline_pvalue": round(float(stat_base.pvalue), 4),
        "wilcoxon_vs_frontier_pvalue": round(float(stat_front.pvalue), 4),
        "verdict_vs_baseline": verdict,
        "label_classes": list(classes),
        "n_wc2026": len(y_wc26),
        "n_historical": len(y_hist),
    }, oof_probs, per_match_losses, classes


def save_artifacts(exp_name, metrics, oof_probs, per_match_losses, classes, feature_list, run_info):
    out_dir = os.path.join(ARTIFACTS_DIR, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    metrics["experiment"] = exp_name

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(f"{out_dir}/oof_probs.npy", oof_probs)

    oof_df = pd.DataFrame(oof_probs, columns=[f"p_{c}" for c in classes])
    oof_df["per_match_log_loss"] = per_match_losses
    oof_df.to_csv(f"{out_dir}/oof_predictions.csv", index=False)

    run_info.update({"features": feature_list, "seed": SEED,
                     "cv": f"RepeatedStratifiedKFold(5x10, seed=0)"})
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


if __name__ == "__main__":
    print("Loading data...")
    df_wc26 = build_match_features(include_context=False)
    y_wc26 = df_wc26["label"].values

    hist_path = "/home/user/research/wave3-context/data/historical_wc_training.csv"
    hist_df = pd.read_csv(hist_path)

    # Load teams for Elo alignment
    teams_df = pd.read_csv(
        "/home/user/research/fifa_extract/wc2026-trees-study-main/fifa_data/teams.csv"
    )
    teams_df["elo_rating"] = pd.to_numeric(teams_df["elo_rating"], errors="coerce")

    # Build running Elo for alignment
    print("Building running Elo for scale alignment...")
    results = pd.read_csv("/home/user/research/data/international_results.csv")
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    results = results.dropna(subset=["home_score", "away_score"])

    final_elo, _, _ = build_running_elo(results)

    # Compute Elo alignment
    print("\nAligning Elo scales...")
    lr_align = compute_elo_alignment(df_wc26, final_elo, teams_df)
    hist_df_aligned = apply_elo_alignment(hist_df, lr_align)

    print(f"\nAligned elo_diff stats:")
    print(f"  WC-2026: mean={df_wc26['elo_diff'].mean():.1f}, std={df_wc26['elo_diff'].std():.1f}")
    print(f"  Hist (aligned): mean={hist_df_aligned['elo_diff'].mean():.1f}, std={hist_df_aligned['elo_diff'].std():.1f}")

    feats = ["elo_diff", "host_advantage"]
    X_wc26 = df_wc26[feats].fillna(0).values
    X_hist_aligned = hist_df_aligned[feats].fillna(0).values

    # --- W3C-04c: Aligned historical augmentation ---
    print("\n--- W3C-04c: Aligned Elo + Historical Augmentation ---")
    best_loss = float("inf")
    best_m, best_oof, best_pml, best_cls, best_cfg = None, None, None, None, {}

    for hist_w in [0.1, 0.2, 0.3, 0.5]:
        for C in [0.3, 1.0, 2.0]:
            m, oof, pml, cls = cv_evaluate_augmented(
                X_wc26, y_wc26, X_hist_aligned, hist_df["label"].values, hist_w,
                lambda C=C: LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver="lbfgs"),
                f"W3C-04c-w{hist_w}-C{C}"
            )
            print(f"  w={hist_w}, C={C}: {m['cv_log_loss_mean']:.4f} ± {m['cv_log_loss_std']:.4f}")
            if m["cv_log_loss_mean"] < best_loss:
                best_loss = m["cv_log_loss_mean"]
                best_m, best_oof, best_pml, best_cls = m, oof, pml, cls
                best_cfg = {"hist_w": hist_w, "C": C}

    best_m["best_config"] = best_cfg
    save_artifacts("W3C-04c", best_m, best_oof, best_pml, best_cls, feats, {
        "experiment": "W3C-04c",
        "description": f"Elo-scale-aligned historical WC 2010-2022 augmentation. Alignment: linear regression on WC-2026 teams anchor. Best: {best_cfg}",
        "model": f"LogisticRegression(C={best_cfg.get('C')})",
        "elo_alignment": {"slope": lr_align.coef_[0], "intercept": lr_align.intercept_},
    })

    # --- W3C-04d: WC-2022 only (most recent, most similar) ---
    print("\n--- W3C-04d: WC-2022 Only Augmentation ---")
    hist_2022 = hist_df_aligned[hist_df_aligned["year"] == 2022].copy()
    print(f"WC-2022 matches: {len(hist_2022)}")

    X_hist_2022 = hist_2022[feats].fillna(0).values
    y_hist_2022 = hist_2022["label"].values

    best_loss2 = float("inf")
    best_m2, best_oof2, best_pml2, best_cls2, best_cfg2 = None, None, None, None, {}

    for hist_w in [0.2, 0.5, 1.0, 2.0]:
        for C in [0.3, 1.0, 2.0]:
            m, oof, pml, cls = cv_evaluate_augmented(
                X_wc26, y_wc26, X_hist_2022, y_hist_2022, hist_w,
                lambda C=C: LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver="lbfgs"),
                f"W3C-04d-w{hist_w}-C{C}"
            )
            print(f"  w={hist_w}, C={C}: {m['cv_log_loss_mean']:.4f}")
            if m["cv_log_loss_mean"] < best_loss2:
                best_loss2 = m["cv_log_loss_mean"]
                best_m2, best_oof2, best_pml2, best_cls2 = m, oof, pml, cls
                best_cfg2 = {"hist_w": hist_w, "C": C}

    best_m2["best_config"] = best_cfg2
    save_artifacts("W3C-04d", best_m2, best_oof2, best_pml2, best_cls2, feats, {
        "experiment": "W3C-04d",
        "description": f"WC-2022 Qatar only augmentation (64 matches, most recent). Best: {best_cfg2}",
        "model": f"LogisticRegression(C={best_cfg2.get('C')})",
        "n_wc2022": len(hist_2022),
    })

    # --- W3C-05: Squad + Aligned Historical ---
    print("\n--- W3C-05: Elo+Squad + Aligned Historical ---")
    all_feats = feats + SQUAD_FEATURES
    X_wc26_sq = df_wc26[all_feats].fillna(0).values
    # Historical: squad features = 0
    X_hist_sq = np.hstack([X_hist_aligned, np.zeros((len(X_hist_aligned), len(SQUAD_FEATURES)))])

    best_loss3 = float("inf")
    best_m3, best_oof3, best_pml3, best_cls3, best_cfg3 = None, None, None, None, {}

    for hist_w in [0.2, 0.3, 0.5]:
        for C in [0.05, 0.1, 0.3]:
            m, oof, pml, cls = cv_evaluate_augmented(
                X_wc26_sq, y_wc26, X_hist_sq, hist_df["label"].values, hist_w,
                lambda C=C: LogisticRegression(C=C, max_iter=1000, random_state=SEED, solver="lbfgs"),
                f"W3C-05-w{hist_w}-C{C}"
            )
            print(f"  w={hist_w}, C={C}: {m['cv_log_loss_mean']:.4f}")
            if m["cv_log_loss_mean"] < best_loss3:
                best_loss3 = m["cv_log_loss_mean"]
                best_m3, best_oof3, best_pml3, best_cls3 = m, oof, pml, cls
                best_cfg3 = {"hist_w": hist_w, "C": C}

    best_m3["best_config"] = best_cfg3
    save_artifacts("W3C-05", best_m3, best_oof3, best_pml3, best_cls3, all_feats, {
        "experiment": "W3C-05",
        "description": f"Elo+squad features + aligned historical WC augmentation (squad=0 for historical). Best: {best_cfg3}",
        "model": f"LogisticRegression(C={best_cfg3.get('C')})",
    })

    print("\nAll done!")
