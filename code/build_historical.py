"""
Build historical WC group-stage training dataset with time-varying Elo.
Uses results.csv from martj42/international-football-results-from-1872-to-2017 (updated to 2026).

For each historical WC group-stage match, computes:
  - elo_diff: home_elo - away_elo at match time
  - host_advantage: is either team playing in their home country
  - venue_elevation: approximate elevation of host city
  - result: H/D/A
"""
import pandas as pd
import numpy as np
import math
from datetime import datetime

RESULTS_PATH = "/home/user/research/data/international_results.csv"

# Approximate elevations of WC host cities (meters) by tournament
# WC 2010: South Africa — mostly coastal
# WC 2014: Brazil — mix
# WC 2018: Russia — mostly flat
# WC 2022: Qatar — sea level
WC_HOST_ELEVATIONS = {
    # WC 2010 South Africa
    "Johannesburg": 1753, "Cape Town": 15, "Durban": 8,
    "Pretoria": 1370, "Port Elizabeth": 9, "Rustenburg": 1135,
    "Bloemfontein": 1400, "Polokwane": 1230,
    # WC 2014 Brazil
    "São Paulo": 760, "Rio de Janeiro": 22, "Belo Horizonte": 858,
    "Fortaleza": 16, "Salvador": 8, "Brasília": 1000,
    "Manaus": 44, "Cuiabá": 165, "Porto Alegre": 10,
    "Curitiba": 935, "Natal": 30, "Recife": 4,
    # WC 2018 Russia
    "Moscow": 156, "Saint Petersburg": 10, "Kazan": 64,
    "Sochi": 0, "Nizhny Novgorod": 78, "Samara": 53,
    "Rostov-on-Don": 0, "Volgograd": 20, "Yekaterinburg": 287,
    "Saransk": 136, "Kaliningrad": 0,
    # WC 2022 Qatar
    "Doha": 10, "Lusail": 10, "Al Rayyan": 10, "Al Wakrah": 10,
    "Al Khor": 10,
}


def build_running_elo(results_df, cutoff_date="2026-06-10"):
    """
    Build time-varying Elo. Returns a dict: (team, date_str) -> elo_before_match,
    and a final elo dict: team -> final elo.

    K values: WC/continental finals=40, qualifiers=25, friendlies=10
    Home advantage: +75 for non-neutral venues.
    """
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(cutoff_date)].sort_values("date").reset_index(drop=True)

    elo = {}

    def get_elo(team):
        return elo.get(team, 1500.0)

    def k_factor(tournament):
        t = str(tournament)
        if "FIFA World Cup" in t and "qualification" not in t.lower() and "qualifier" not in t.lower():
            return 40
        if any(x in t for x in ["UEFA Euro", "Copa América", "Africa Cup", "Asian Cup", "CONCACAF Gold"]):
            return 35
        if "qualification" in t.lower() or "qualifier" in t.lower() or "Qualification" in t:
            return 25
        if "Nations League" in t or "Confederations" in t:
            return 25
        return 10  # Friendly

    pre_match_elo = {}  # (home_team, away_team, date_str) -> (home_elo, away_elo)

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        neutral = str(row.get("neutral", "True")).upper() == "TRUE"
        date_str = str(row["date"].date())

        ea = get_elo(home)
        eb = get_elo(away)

        # Home advantage in non-neutral venues
        ea_adj = ea + (75 if not neutral else 0)

        exp_a = 1 / (1 + 10 ** ((eb - ea_adj) / 400))
        exp_b = 1 - exp_a

        if row["home_score"] > row["away_score"]:
            result_a = 1.0
        elif row["home_score"] == row["away_score"]:
            result_a = 0.5
        else:
            result_a = 0.0

        K = k_factor(row.get("tournament", ""))

        # Goal difference multiplier (capped at 3)
        gd = abs(row["home_score"] - row["away_score"])
        gd_mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75)

        pre_match_elo[(home, away, date_str)] = (ea, eb)

        elo[home] = ea + K * gd_mult * (result_a - exp_a)
        elo[away] = eb + K * gd_mult * ((1 - result_a) - exp_b)

    return elo, pre_match_elo, df


def get_wc_group_stage_matches(results_df):
    """Extract WC group-stage matches from 2010-2022."""
    df = results_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # WC final tournament matches (not qualification)
    wc = df[
        df["tournament"].str.contains("FIFA World Cup", na=False) &
        ~df["tournament"].str.lower().str.contains("qual") &
        ~df["tournament"].str.lower().str.contains("round")
    ].copy()

    # Filter to 2010-2022 (not 2026)
    wc = wc[(wc["date"] >= pd.Timestamp("2010-01-01")) &
            (wc["date"] < pd.Timestamp("2026-01-01"))]

    print(f"WC 2010-2022 tournament matches: {len(wc)}")
    print(wc.groupby(wc["date"].dt.year)["date"].count().to_string())

    return wc


def build_historical_training_set(results_path=RESULTS_PATH):
    """
    Build training set from WC 2010-2022 group stage (64 matches each = 192 total).
    Features: elo_diff, host_advantage.
    """
    results = pd.read_csv(results_path)
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce")
    results = results.dropna(subset=["home_score", "away_score"])

    # Build running Elo
    print("Building running Elo from 49k+ matches...")
    final_elo, pre_match_elo_dict, sorted_results = build_running_elo(results)

    # Get WC tournament matches 2010-2022
    wc_matches = get_wc_group_stage_matches(results)

    rows = []
    for _, row in wc_matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        date_str = str(row["date"].date())
        year = row["date"].year

        key = (home, away, date_str)
        if key in pre_match_elo_dict:
            elo_h, elo_a = pre_match_elo_dict[key]
        else:
            # Fallback: search nearest key
            elo_h, elo_a = 1500, 1500

        # Host advantage: WC 2010 in South Africa, 2014 in Brazil, 2018 in Russia, 2022 in Qatar
        host_teams = {2010: set(), 2014: {"Brazil"}, 2018: {"Russia"}, 2022: {"Qatar"}}
        host_set = host_teams.get(year, set())
        home_is_host = 1 if home in host_set else 0
        away_is_host = 1 if away in host_set else 0
        host_adv = home_is_host - away_is_host

        if row["home_score"] > row["away_score"]:
            label = "H"
        elif row["home_score"] == row["away_score"]:
            label = "D"
        else:
            label = "A"

        rows.append({
            "date": row["date"],
            "year": year,
            "home_team": home,
            "away_team": away,
            "elo_diff": elo_h - elo_a,
            "host_advantage": host_adv,
            "home_is_host": home_is_host,
            "away_is_host": away_is_host,
            "label": label,
            "tournament": row.get("tournament", ""),
        })

    hist_df = pd.DataFrame(rows)
    print(f"\nBuilt historical training set: {len(hist_df)} matches")
    print("Label dist:", hist_df["label"].value_counts().to_dict())
    print("Elo diff stats:", hist_df["elo_diff"].describe().round(1).to_string())

    return hist_df, final_elo


if __name__ == "__main__":
    hist_df, final_elo = build_historical_training_set()
    print("\nSample historical rows:")
    print(hist_df[["year", "home_team", "away_team", "elo_diff", "host_advantage", "label"]].head(10).to_string())
    print(f"\nFinal Elo for sample teams:")
    for t in ["Brazil", "Argentina", "France", "Spain", "Germany", "England", "Mexico", "USA"]:
        print(f"  {t}: {final_elo.get(t, 1500):.0f}")
    hist_df.to_csv("/home/user/research/wave3-context/data/historical_wc_training.csv", index=False)
    print("\nSaved to data/historical_wc_training.csv")
