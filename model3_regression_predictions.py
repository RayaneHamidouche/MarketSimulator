import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

stocks = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

VOLATILITY_WINDOW = 20
VOLUME_WINDOW = 20
TRAIN_SIZE = 0.8


# Create folder for predictions
os.makedirs("Predictions", exist_ok=True)


# Store all predictions
all_predictions = []


# --------------------------------------------------
# PROCESS EACH STOCK
# --------------------------------------------------

for stock in stocks:

    print(f"Processing {stock}...")

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    df = pd.read_csv(
        f"C:/Users/heatw/OneDrive - Twyford Academies/Documents/MarketData/{stock}.csv",
        index_col=0
    )

    # Convert dates
    df.index = pd.to_datetime(
        df.index,
        dayfirst=True
    )

    # Sort chronologically
    df = df.sort_index()

    # Make sure numerical columns are numeric
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])


    # --------------------------------------------------
    # CALCULATE RETURNS
    # --------------------------------------------------

    df["return"] = df["close"].pct_change()

    # Absolute return
    df["absolute_return"] = df["return"].abs()


    # --------------------------------------------------
    # CALCULATE VOLATILITY
    # --------------------------------------------------

    df["volatility"] = (
        df["return"]
        .rolling(VOLATILITY_WINDOW)
        .std()
    )


    # --------------------------------------------------
    # CALCULATE RELATIVE VOLUME
    # --------------------------------------------------

    df["average_volume"] = (
        df["volume"]
        .rolling(VOLUME_WINDOW)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["average_volume"]
    )


    # --------------------------------------------------
    # CREATE LAGGED FEATURES
    #
    # We shift everything by one day so that the model
    # only uses information that was available BEFORE
    # the volatility we are trying to predict.
    # --------------------------------------------------

    df["volatility_lag1"] = (
        df["volatility"].shift(1)
    )

    df["absolute_return_lag1"] = (
        df["absolute_return"].shift(1)
    )

    df["volume_ratio_lag1"] = (
        df["volume_ratio"].shift(1)
    )


    # --------------------------------------------------
    # REMOVE MISSING VALUES
    # --------------------------------------------------

    df = df.dropna(
        subset=[
            "volatility",
            "volatility_lag1",
            "absolute_return_lag1",
            "volume_ratio_lag1"
        ]
    )


    # --------------------------------------------------
    # DEFINE FEATURES AND TARGET
    # --------------------------------------------------

    features = [
        "volatility_lag1",
        "absolute_return_lag1",
        "volume_ratio_lag1"
    ]

    X = df[features]

    y = df["volatility"]


    # --------------------------------------------------
    # TRAIN / TEST SPLIT
    #
    # IMPORTANT:
    # We DO NOT randomly shuffle financial data.
    # The first 80% is training data.
    # The final 20% is testing data.
    # --------------------------------------------------

    split_index = int(len(df) * TRAIN_SIZE)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]


    # --------------------------------------------------
    # TRAIN REGRESSION MODEL
    # --------------------------------------------------

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------
    # MAKE PREDICTIONS
    # --------------------------------------------------

    predictions = model.predict(X_test)


    # --------------------------------------------------
    # CALCULATE ABSOLUTE ERROR
    # --------------------------------------------------

    absolute_errors = (
        np.abs(predictions - y_test)
    )


    # --------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------

    for i in range(len(X_test)):

        all_predictions.append({
            "date": X_test.index[i],
            "stock": stock,
            "predicted_volatility": predictions[i],
            "actual_volatility": y_test.iloc[i],
            "absolute_error": absolute_errors.iloc[i]
        })


    # --------------------------------------------------
    # DISPLAY MODEL INFORMATION
    # --------------------------------------------------

    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))

    print("Coefficients:")

    for feature, coefficient in zip(
        features,
        model.coef_
    ):
        print(
            f"  {feature}: {coefficient}"
        )

    print(
        "Intercept:",
        model.intercept_
    )

    print()


# --------------------------------------------------
# CREATE FINAL PREDICTION DATAFRAME
# --------------------------------------------------

all_predictions = pd.DataFrame(
    all_predictions
)


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

all_predictions.to_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model3_regression_predictions.csv",
    index=False
)


# --------------------------------------------------
# CALCULATE MAE
# --------------------------------------------------

mae_results = []

for stock in stocks:

    stock_data = all_predictions[
        all_predictions["stock"] == stock
    ]

    mae = stock_data["absolute_error"].mean()

    mae_results.append({
        "stock": stock,
        "MAE": mae
    })


mae_results = pd.DataFrame(
    mae_results
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\nModel 3 - Mean Absolute Error:")
print(mae_results)


# --------------------------------------------------
# SAVE MAE
# --------------------------------------------------

mae_results.to_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model3_mae.csv",
    index=False
)


print("\nFinished!")

print(
    "Predictions saved to:"
)

print(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model3_regression_predictions.csv"
)

print("\nMAE saved to:")

print(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model3_mae.csv"
)