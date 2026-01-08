# F1 Race Winner Prediction

This project uses machine learning to predict Formula 1 race winners based on historical data from the OpenF1 API.

## Features

- Data collection from OpenF1 API
- Feature engineering from race data, pit stops, car telemetry, etc.
- ML model training for win prediction
- Prediction for a given driver
- Simulations to estimate win probabilities

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Run data collection: `python src/data_collection.py`
3. Preprocess data: `python src/preprocessing.py`
4. Train model: `python src/model.py`
5. Predict: `python src/predict.py --driver "Max Verstappen"`

## Usage

Input a driver's name to get their predicted win probability.
