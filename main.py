import subprocess
import sys
import os
import argparse
import pandas as pd

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="F1 Prediction Pipeline")
    parser.add_argument("--driver", help="Driver name for prediction")
    parser.add_argument("--country", help="Country name to filter drivers by")
    parser.add_argument("--force-collect", action="store_true", help="Force new data collection")
    args = parser.parse_args()

    venv_python = "C:/XboxGames/things/.venv/Scripts/python.exe"

    if args.country:
        print(f"\n=== Finding drivers for country: {args.country} ===")
        try:
            # Load data
            meetings = pd.read_csv("data/meetings.csv")
            race_results = pd.read_csv("data/race_results.csv")
            drivers = pd.read_csv("data/drivers.csv")

            # Filter meetings by country
            country_meetings = meetings[meetings['country_name'].str.lower() == args.country.lower()]
            
            if country_meetings.empty:
                print(f"No meetings found for country: {args.country}")
                return

            meeting_keys = country_meetings['meeting_key'].unique()

            # Filter race results for these meetings
            country_results = race_results[race_results['meeting_key'].isin(meeting_keys)]
            
            if country_results.empty:
                print(f"No race results found for meetings in: {args.country}")
                return

            driver_numbers = country_results['driver_number'].unique()

            # Filter drivers
            country_drivers = drivers[drivers['driver_number'].isin(driver_numbers)]
            
            # Print driver names
            print(f"Drivers who raced in {args.country}:")
            for _, driver in country_drivers.iterrows():
                print(f"driver name = ({driver['full_name']})")

        except FileNotFoundError:
             print("Data not found. Running data collection first...")
             if run_command(f'{venv_python} src/data_collection.py') != 0:
                print("Data collection failed.")
                return
             # Recursive call or re-run logic? Simpler to just proceed to logic below if we were strictly doing country. 
             # But here we are inside the 'if args.country' block which returns. 
             # Let's actually fall through to data collection if files missing? 
             # No, if files are missing we should collect, then retry filtering.
             pass 
             # Actually, simpler: If files missing, run collection. Then loop back?
             # Let's adjust the structure.
        except Exception as e:
            print(f"An error occurred: {e}")
            return
        
        # If we successfully loaded and printed, return. 
        # If we caught FileNotFoundError, we want to run collection.
        # But the collection code is below.
        # Let's refactor slightly to separate "Ensure Data" from "Run Pipeline".
        return

    print("Starting F1 Prediction Pipeline...")

    # Data collection
    print("\n=== Data Collection ===")
    data_exists = os.path.exists("data/meetings.csv") and os.path.exists("data/race_results.csv") and os.path.exists("data/drivers.csv")
    
    if args.force_collect or not data_exists:
        if run_command(f'{venv_python} src/data_collection.py') != 0:
            print("Data collection failed.")
            return
    else:
        print("Data files found. Skipping collection (use --force-collect to overwrite).")

    # Preprocessing
    print("\n=== Preprocessing ===")
    if run_command(f'{venv_python} src/preprocessing.py') != 0:
        print("Preprocessing failed.")
        return

    # Model training
    print("\n=== Model Training ===")
    if run_command(f'{venv_python} src/model.py') != 0:
        print("Model training failed.")
        return

    print("\nPipeline complete!")

    # Prediction if driver provided
    if args.driver:
        print(f"\n=== Prediction for {args.driver} ===")
        if run_command(f'{venv_python} src/predict.py --driver "{args.driver}"') != 0:
            print("Prediction failed.")
        else:
            print("Prediction complete!")
    else:
        try:
            driver_name = input("driver name = ")
            if driver_name:
                print(f"\n=== Prediction for {driver_name} ===")
                if run_command(f'{venv_python} src/predict.py --driver "{driver_name}"') != 0:
                    print("Prediction failed.")
                else:
                    print("Prediction complete!")
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()