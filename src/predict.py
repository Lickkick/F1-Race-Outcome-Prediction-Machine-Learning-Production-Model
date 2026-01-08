import pandas as pd
import joblib
import argparse
import numpy as np

def predict_win_probability(driver_name, simulations=1000):
    # Load model
    model = joblib.load("models/win_predictor.pkl")

    # Load driver stats
    driver_stats = pd.read_csv("data/driver_stats.csv")

    # Find driver
    driver = driver_stats[driver_stats["full_name"].str.contains(driver_name, case=False)]
    if driver.empty:
        print("Driver not found.")
        return

    # Get features
    features = ["avg_qual_pos", "avg_pit_count", "avg_pit_time", "avg_stint_count", "total_races", "total_wins", "team_encoded"]
    X = driver[features].values[0]

    # Run simulations with noise
    probs = []
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")

    for _ in range(simulations):
        # Add small noise to features
        noise = np.random.normal(0, 0.1, len(X))
        X_sim = X + noise
        
        # Create DataFrame to avoid feature name warnings
        X_df = pd.DataFrame([X_sim], columns=features)
        
        prob = model.predict_proba(X_df)[0][1]
        probs.append(prob)

    avg_prob = np.mean(probs)
    print(f"After running various simulations, the given percentage is {avg_prob:.2%}")
    print(f"The driver {driver['full_name'].values[0]} has a win probability of {avg_prob:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True, help="Driver name")
    parser.add_argument("--simulations", type=int, default=1000, help="Number of simulations")
    args = parser.parse_args()
    predict_win_probability(args.driver, args.simulations)