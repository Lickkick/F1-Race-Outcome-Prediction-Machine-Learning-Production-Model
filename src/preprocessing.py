import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("Preprocessing script started")

def preprocess_data():
    print("Loading data...")
    # Load data
    results = pd.read_csv("data/race_results.csv")
    drivers = pd.read_csv("data/drivers.csv")
    pits = pd.read_csv("data/pit_stops.csv")
    stints = pd.read_csv("data/stints.csv")
    try:
        starting_grid = pd.read_csv("data/starting_grid.csv")
        has_starting_grid = True
        print("Starting grid data loaded.")
    except pd.errors.EmptyDataError:
        has_starting_grid = False
        starting_grid = pd.DataFrame()
        print("Starting grid data not available.")

    print("Merging results with drivers...")
    # Merge results with drivers
    results = results.merge(drivers[["driver_number", "full_name", "team_name", "meeting_key", "session_key"]], on=["driver_number", "meeting_key", "session_key"], how="left")

    if has_starting_grid:
        print("Merging with starting grid...")
        # Merge with starting grid for qual_pos
        starting_grid = starting_grid.rename(columns={"position": "qual_pos"})
        results = results.merge(starting_grid[["session_key", "driver_number", "qual_pos"]], on=["session_key", "driver_number"], how="left")
    else:
        results["qual_pos"] = np.nan

    print("Computing pit features...")
    # Compute pit features per driver per session
    pit_features = pits.groupby(["session_key", "driver_number"]).agg(
        pit_count=("pit_duration", "count"),
        pit_avg_time=("pit_duration", "mean")
    ).reset_index()

    print("Computing stint features...")
    # Stint features
    stint_features = stints.groupby(["session_key", "driver_number"]).agg(
        stint_count=("stint_number", "count"),
    ).reset_index()

    print("Merging features...")
    # Merge all
    df = results.merge(pit_features, on=["session_key", "driver_number"], how="left").fillna(0)
    df = df.merge(stint_features, on=["session_key", "driver_number"], how="left").fillna(0)

    print("Encoding teams...")
    # Encode team
    le = LabelEncoder()
    df["team_encoded"] = le.fit_transform(df["team_name"].fillna("Unknown"))

    # Target: win
    df["win"] = (df["position"] == 1).astype(int)

    print("Computing driver stats...")
    # For each driver, compute averages
    driver_stats = df.groupby("driver_number").agg(
        avg_qual_pos=("qual_pos", "mean"),
        avg_pit_count=("pit_count", "mean"),
        avg_pit_time=("pit_avg_time", "mean"),
        avg_stint_count=("stint_count", "mean"),
        total_races=("win", "count"),
        total_wins=("win", "sum"),
        team_encoded=("team_encoded", "first"),
        full_name=("full_name", "first")
    ).reset_index()

    # For dataset, for each race, use driver stats as features
    df = df.merge(driver_stats, on="driver_number", suffixes=("", "_avg"))

    # Features
    features = ["avg_qual_pos", "avg_pit_count", "avg_pit_time", "avg_stint_count", "total_races", "total_wins", "team_encoded"]
    X = df[features]
    y = df["win"]

    print("Saving processed data...")
    # Save processed data
    df.to_csv("data/processed_data.csv", index=False)
    driver_stats.to_csv("data/driver_stats.csv", index=False)

    print("Preprocessing complete.")
    return X, y, driver_stats

if __name__ == "__main__":
    preprocess_data()