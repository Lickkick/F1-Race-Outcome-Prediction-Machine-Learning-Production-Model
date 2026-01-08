import requests
import pandas as pd
import time
import os

print("Script started")

BASE_URL = "https://api.openf1.org/v1"

def fetch_data(endpoint, params=None):
    url = f"{BASE_URL}/{endpoint}"
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def collect_data():
    print("Fetching meetings...")
    # Get meetings for 2022-2025
    meetings = []
    for year in range(2020, 2026):
        meetings.extend(fetch_data("meetings", {"year": year}))
    meetings_df = pd.DataFrame(meetings)
    meetings_df.to_csv("data/meetings.csv", index=False)

    print("Fetching sessions...")
    # Get sessions
    sessions = []
    for meeting in meetings:
        sess = fetch_data("sessions", {"meeting_key": meeting["meeting_key"]})
        sessions.extend(sess)
        time.sleep(0.1)  # Rate limit
    sessions_df = pd.DataFrame(sessions)
    sessions_df.to_csv("data/sessions.csv", index=False)

    # Get race sessions
    race_sessions = sessions_df[sessions_df["session_name"] == "Race"]

    print("Fetching race results...")
    # Get session results
    results = []
    for _, session in race_sessions.iterrows():
        res = fetch_data("session_result", {"session_key": session["session_key"]})
        for r in res:
            r["meeting_key"] = session["meeting_key"]
            r["year"] = session["year"]
        results.extend(res)
        time.sleep(0.1)
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/race_results.csv", index=False)

    print("Fetching pit stops...")
    # Get pit stops
    pits = []
    for _, session in race_sessions.iterrows():
        pit = fetch_data("pit", {"session_key": session["session_key"]})
        pits.extend(pit)
        time.sleep(0.1)
    pits_df = pd.DataFrame(pits)
    pits_df.to_csv("data/pit_stops.csv", index=False)

    print("Fetching stints...")
    # Get stints
    stints = []
    for _, session in race_sessions.iterrows():
        stint = fetch_data("stints", {"session_key": session["session_key"]})
        stints.extend(stint)
        time.sleep(0.1)
    stints_df = pd.DataFrame(stints)
    stints_df.to_csv("data/stints.csv", index=False)

    print("Fetching starting grid...")
    # Get starting grid
    starting_grids = []
    for _, session in race_sessions.iterrows():
        sg = fetch_data("starting_grid", {"session_key": session["session_key"]})
        starting_grids.extend(sg)
        time.sleep(0.1)
    pd.DataFrame(starting_grids).to_csv("data/starting_grid.csv", index=False)

    print("Fetching drivers...")
    # Get drivers
    drivers = fetch_data("drivers")
    drivers_df = pd.DataFrame(drivers)
    drivers_df.to_csv("data/drivers.csv", index=False)

    print("Data collection complete.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    collect_data()