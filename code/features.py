"""Wave-3 feature engineering: context/fatigue features + squad features."""
import pandas as pd
import numpy as np
from datetime import datetime
import math

HOSTS = {"USA", "MEX", "CAN"}
DATA_DIR = "/home/user/research/fifa_extract/wc2026-trees-study-main/fifa_data"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def load_data(data_dir=DATA_DIR):
    matches = pd.read_csv(f"{data_dir}/matches_detailed.csv")
    teams = pd.read_csv(f"{data_dir}/teams.csv")
    squads = pd.read_csv(f"{data_dir}/squads_and_players.csv")
    venues = pd.read_csv(f"{data_dir}/venues.csv")
    return matches, teams, squads, venues


def compute_squad_features(squads, teams):
    """Aggregate per-team squad features (pre-match only)."""
    squad = squads.merge(teams[["team_id", "fifa_code"]], on="team_id", how="left")
    ref = datetime(2026, 6, 11)
    squad["dob"] = pd.to_datetime(squad["date_of_birth"], errors="coerce")
    squad["age"] = (ref - squad["dob"]).dt.days / 365.25
    squad["mv"] = pd.to_numeric(squad["market_value_eur"], errors="coerce").fillna(0)
    squad["goals_val"] = pd.to_numeric(squad["goals"], errors="coerce").fillna(0)
    squad["caps_val"] = pd.to_numeric(squad["caps"], errors="coerce").fillna(0)
    squad["height_val"] = pd.to_numeric(squad["height_cm"], errors="coerce")

    feats = squad.groupby("fifa_code").agg(
        squad_total_mv=("mv", "sum"),
        squad_top11_mv=("mv", lambda x: x.nlargest(11).sum()),
        squad_mean_age=("age", "mean"),
        squad_mean_caps=("caps_val", "mean"),
        squad_total_goals=("goals_val", "sum"),
        squad_mean_height=("height_val", "mean"),
    ).reset_index()

    gk = squad[squad["position"] == "GK"].groupby("fifa_code")["mv"].sum().rename("gk_mv")
    feats = feats.merge(gk, on="fifa_code", how="left")
    feats["gk_mv"] = feats["gk_mv"].fillna(0)

    att = squad[squad["position"].isin(["FW"])].groupby("fifa_code")["goals_val"].sum().rename("att_goals")
    feats = feats.merge(att, on="fifa_code", how="left")
    feats["att_goals"] = feats["att_goals"].fillna(0)

    return feats


def compute_context_features(df_completed, venues_df):
    """
    Compute context/fatigue features for each match:
    - rest_days_home / rest_days_away: days since previous match
    - travel_km_home / travel_km_away: great-circle km from previous venue
    - venue_elevation: current venue altitude (m)
    - altitude_diff_home / altitude_diff_away: vs previous venue altitude
    - kickoff_hour_local: local hour approximated from UTC + timezone
    - match_number_in_group: 1st/2nd/3rd group match (stakes)
    """
    df = df_completed.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Venue lookup (by stadium_name)
    venue_map = venues_df.set_index("stadium_name")[["latitude", "longitude", "elevation_meters"]].to_dict("index")

    # Build per-team match history: for each match, what was the previous match?
    # We need to track each team's previous match date + venue
    team_prev = {}  # team_fifa_code -> {date, stadium, lat, lon, elevation}

    rest_home, rest_away = [], []
    travel_home, travel_away = [], []
    elev_home, elev_away = [], []
    alt_diff_home, alt_diff_away = [], []

    for _, row in df.iterrows():
        home_team = row["home_fifa_code"]
        away_team = row["away_fifa_code"]
        match_date = row["date"]
        stadium = row["stadium_name"]
        vinfo = venue_map.get(stadium, {"latitude": 0, "longitude": 0, "elevation_meters": 0})
        cur_lat = vinfo["latitude"]
        cur_lon = vinfo["longitude"]
        cur_elev = vinfo.get("elevation_meters", 0) or 0

        # Home team context
        if home_team in team_prev:
            prev = team_prev[home_team]
            r_h = (match_date - prev["date"]).days
            t_h = haversine_km(prev["lat"], prev["lon"], cur_lat, cur_lon)
            ad_h = cur_elev - prev["elevation"]
        else:
            r_h = 14  # tournament start: give a default large rest
            t_h = 0.0
            ad_h = 0.0

        # Away team context
        if away_team in team_prev:
            prev = team_prev[away_team]
            r_a = (match_date - prev["date"]).days
            t_a = haversine_km(prev["lat"], prev["lon"], cur_lat, cur_lon)
            ad_a = cur_elev - prev["elevation"]
        else:
            r_a = 14
            t_a = 0.0
            ad_a = 0.0

        rest_home.append(r_h)
        rest_away.append(r_a)
        travel_home.append(t_h)
        travel_away.append(t_a)
        elev_home.append(cur_elev)
        elev_away.append(cur_elev)
        alt_diff_home.append(ad_h)
        alt_diff_away.append(ad_a)

        # Update team_prev AFTER reading (for next match)
        team_prev[home_team] = {"date": match_date, "lat": cur_lat, "lon": cur_lon, "elevation": cur_elev}
        team_prev[away_team] = {"date": match_date, "lat": cur_lat, "lon": cur_lon, "elevation": cur_elev}

    df["rest_days_home"] = rest_home
    df["rest_days_away"] = rest_away
    df["travel_km_home"] = travel_home
    df["travel_km_away"] = travel_away
    df["venue_elevation"] = elev_home
    df["alt_diff_home"] = alt_diff_home
    df["alt_diff_away"] = alt_diff_away

    # Rest advantage (positive = home has more rest)
    df["rest_diff"] = df["rest_days_home"] - df["rest_days_away"]
    # Travel advantage (positive = away traveled more)
    df["travel_diff"] = df["travel_km_away"] - df["travel_km_home"]

    # Kickoff local hour: approximate UTC to local via longitude (1hr per 15 degrees)
    df["kickoff_utc_hour"] = pd.to_numeric(df["kickoff_time_utc"].str.split(":").str[0], errors="coerce").fillna(20)
    # Merge venue longitude
    lon_map = venues_df.set_index("stadium_name")["longitude"].to_dict()
    df["venue_lon"] = df["stadium_name"].map(lon_map).fillna(-90)
    df["timezone_offset"] = (df["venue_lon"] / 15.0).round()
    df["kickoff_local_hour"] = ((df["kickoff_utc_hour"] + df["timezone_offset"]) % 24)

    # Match number in group (1st, 2nd, 3rd game) per team
    # Sort by date per team
    home_match_num = df.groupby("home_fifa_code").cumcount() + 1
    away_match_num = df.groupby("away_fifa_code").cumcount() + 1
    df["match_num_home"] = home_match_num
    df["match_num_away"] = away_match_num
    # If 3rd game (match_num==3), potentially dead-rubber — encode as stake_flag
    # Actually encode as: match_number (1,2,3) = increasing stakes before elimination
    df["match_num_diff"] = df["match_num_home"] - df["match_num_away"]  # usually 0

    # Host at home-country venue
    # MEX hosts in MEX cities, USA hosts in USA cities, CAN hosts in CAN cities
    country_map = venues_df.set_index("stadium_name")["country"].to_dict()
    df["venue_country"] = df["stadium_name"].map(country_map)
    df["home_host_home_venue"] = (
        (df["home_fifa_code"].isin(HOSTS)) &
        (df["venue_country"] == df["home_fifa_code"])
    ).astype(int)
    df["away_host_home_venue"] = (
        (df["away_fifa_code"].isin(HOSTS)) &
        (df["venue_country"] == df["away_fifa_code"])
    ).astype(int)

    return df


def build_match_features(include_context=True, data_dir=DATA_DIR):
    """Build the full pre-match feature matrix for the 64 WC-2026 completed matches."""
    matches, teams, squads, venues = load_data(data_dir)

    df = matches[matches["status"] == "Completed"].copy()
    assert len(df) == 64, f"Expected 64 completed, got {len(df)}"

    # Target
    df["label"] = np.where(df["home_score"] > df["away_score"], "H",
                   np.where(df["home_score"] == df["away_score"], "D", "A"))

    # Squad features
    sq_feats = compute_squad_features(squads, teams)

    # Team info
    team_info = teams[["fifa_code", "elo_rating", "fifa_ranking_pre_tournament", "confederation"]].copy()
    team_info["elo_rating"] = pd.to_numeric(team_info["elo_rating"], errors="coerce")
    team_info["fifa_ranking_pre_tournament"] = pd.to_numeric(team_info["fifa_ranking_pre_tournament"], errors="coerce")
    team_info = team_info.merge(sq_feats, on="fifa_code", how="left")

    # Venue features
    ven = venues[["stadium_name", "capacity", "elevation_meters"]].copy()
    ven["capacity"] = pd.to_numeric(ven["capacity"], errors="coerce")
    ven["elevation_meters"] = pd.to_numeric(ven["elevation_meters"], errors="coerce").fillna(0)
    df = df.merge(ven, on="stadium_name", how="left")

    # Merge home team info
    home_ti = team_info.rename(columns={c: f"home_{c}" for c in team_info.columns if c != "fifa_code"})
    df = df.merge(home_ti, left_on="home_fifa_code", right_on="fifa_code", how="left").drop(columns=["fifa_code"])

    # Merge away team info
    away_ti = team_info.rename(columns={c: f"away_{c}" for c in team_info.columns if c != "fifa_code"})
    df = df.merge(away_ti, left_on="away_fifa_code", right_on="fifa_code", how="left").drop(columns=["fifa_code"])

    # Difference features
    df["elo_diff"] = df["home_elo_rating"] - df["away_elo_rating"]
    df["rank_diff"] = -(df["home_fifa_ranking_pre_tournament"] - df["away_fifa_ranking_pre_tournament"])  # positive = home ranked higher
    df["mv_top11_diff"] = np.log1p(df["home_squad_top11_mv"]) - np.log1p(df["away_squad_top11_mv"])
    df["caps_diff"] = df["home_squad_mean_caps"] - df["away_squad_mean_caps"]
    df["age_diff"] = df["home_squad_mean_age"] - df["away_squad_mean_age"]
    df["gk_mv_diff"] = np.log1p(df["home_gk_mv"]) - np.log1p(df["away_gk_mv"])
    df["att_goals_diff"] = df["home_att_goals"] - df["away_att_goals"]

    # Host flags
    df["home_is_host"] = df["home_fifa_code"].isin(HOSTS).astype(int)
    df["away_is_host"] = df["away_fifa_code"].isin(HOSTS).astype(int)
    df["host_advantage"] = df["home_is_host"] - df["away_is_host"]

    # Stage encoding
    stage_order = {"Group Stage": 0, "Round of 16": 1, "Quarter-final": 2,
                   "Semi-final": 3, "Third-place play-off": 3, "Final": 4}
    df["stage_enc"] = df["stage_name"].map(stage_order).fillna(0)

    # Context features
    if include_context:
        df = compute_context_features(df, venues)

    return df


CONTEXT_FEATURES = [
    "rest_diff", "travel_diff", "venue_elevation",
    "kickoff_local_hour", "match_num_home", "match_num_away",
    "home_host_home_venue", "away_host_home_venue",
]

ELO_FEATURES = ["elo_diff", "host_advantage"]

SQUAD_FEATURES = [
    "mv_top11_diff", "caps_diff", "age_diff", "gk_mv_diff", "att_goals_diff", "rank_diff"
]

ALL_FEATURES = ELO_FEATURES + SQUAD_FEATURES + CONTEXT_FEATURES


if __name__ == "__main__":
    df = build_match_features()
    print("Shape:", df.shape)
    print("Label dist:", df["label"].value_counts().to_dict())
    print("\nContext features sample:")
    print(df[["home_team_name", "away_team_name", "date", "rest_days_home", "rest_days_away",
              "travel_km_home", "travel_km_away", "venue_elevation", "kickoff_local_hour"]].head(10).to_string())
