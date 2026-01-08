import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model():
    # Load processed data
    df = pd.read_csv("data/processed_data.csv")
    features = ["avg_qual_pos", "avg_pit_count", "avg_pit_time", "avg_stint_count", "total_races", "total_wins", "team_encoded"]
    X = df[features]
    y = df["win"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "models/win_predictor.pkl")
    print("Model saved.")

    return model

if __name__ == "__main__":
    train_model()