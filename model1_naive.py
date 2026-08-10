import pandas as pd
import numpy as np
import os


# ==================================================
# SETTINGS
# ==================================================

stocks = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

VOLATILITY_WINDOW = 20
TRAIN_SIZE = 0.8

os.makedirs("Predictions", exist_ok=True)


# ==================================================
# PART 1: CREATE PREDICTIONS
# ==================================================

all_predictions = []


for stock in stocks:

    print(f"Processing {stock}...")

    # ----------------------------------------------
    # Load data
    # ----------------------------------------------

    df = pd.read_csv(
        f"C:/Users/heatw/OneDrive - Twyford Academies/Documents/MarketData/{stock}.csv",
        index_col=0
    )

    # Convert dates
    df.index = pd.to_datetime(
        df.index,
        dayfirst=True
    )

    # Sort oldest -> newest
    df = df.sort_index()

    # Convert close to numbers
    df["close"] = pd.to_numeric(df["close"])

    # ----------------------------------------------
    # Calculate daily returns
    # ----------------------------------------------

    df["return"] = df["close"].pct_change()

    # ----------------------------------------------
    # Calculate 20-day volatility
    # ----------------------------------------------

    df["volatility"] = (
        df["return"]
        .rolling(VOLATILITY_WINDOW)
        .std()
    )

    # Remove missing volatility values
    df = df.dropna(
        subset=["volatility"]
    )

    # ----------------------------------------------
    # Split into training and test periods
    #
    # Model 1 doesn't actually need training,
    # but we use the same test period as the
    # other models for a fair comparison.
    # ----------------------------------------------

    split_index = int(
        len(df) * TRAIN_SIZE
    )

    test = df.iloc[split_index:].copy()

    # ----------------------------------------------
    # Naïve prediction
    #
    # Tomorrow's volatility =
    # today's volatility
    # ----------------------------------------------

    test["predicted_volatility"] = (
        test["volatility"].shift(1)
    )

    # Remove first test observation because
    # there is no previous volatility within test
    test = test.dropna(
        subset=["predicted_volatility"]
    )

    # ----------------------------------------------
    # Calculate errors
    # ----------------------------------------------

    test["error"] = (
        test["predicted_volatility"]
        - test["volatility"]
    )

    test["absolute_error"] = (
        test["error"].abs()
    )

    test["squared_error"] = (
        test["error"] ** 2
    )

    # ----------------------------------------------
    # Store predictions
    # ----------------------------------------------

    for date, row in test.iterrows():

        all_predictions.append({
            "date": date,
            "stock": stock,
            "predicted_volatility":
                row["predicted_volatility"],
            "actual_volatility":
                row["volatility"],
            "error":
                row["error"],
            "absolute_error":
                row["absolute_error"],
            "squared_error":
                row["squared_error"]
        })


# ==================================================
# PART 2: SAVE PREDICTIONS
# ==================================================

predictions = pd.DataFrame(
    all_predictions
)

predictions.to_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model1_naive_predictions.csv",
    index=False
)

print("\nPredictions saved.")


# ==================================================
# PART 3: LOAD PREDICTIONS
# ==================================================

predictions = pd.read_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model1_naive_predictions.csv"
)


# ==================================================
# PART 4: STATISTICAL ANALYSIS
# ==================================================

results = []


for stock in stocks:

    stock_data = predictions[
        predictions["stock"] == stock
    ].copy()

    # ----------------------------------------------
    # MAE
    # ----------------------------------------------

    mae = stock_data[
        "absolute_error"
    ].mean()

    # ----------------------------------------------
    # RMSE
    # ----------------------------------------------

    rmse = np.sqrt(
        stock_data[
            "squared_error"
        ].mean()
    )

    # ----------------------------------------------
    # Mean error / bias
    #
    # Positive = tends to overpredict
    # Negative = tends to underpredict
    # ----------------------------------------------

    mean_error = stock_data[
        "error"
    ].mean()

    # ----------------------------------------------
    # Mean actual volatility
    # ----------------------------------------------

    mean_actual = stock_data[
        "actual_volatility"
    ].mean()

    # ----------------------------------------------
    # Mean predicted volatility
    # ----------------------------------------------

    mean_predicted = stock_data[
        "predicted_volatility"
    ].mean()

    # ----------------------------------------------
    # Correlation between prediction and reality
    # ----------------------------------------------

    correlation = stock_data[
        "predicted_volatility"
    ].corr(
        stock_data["actual_volatility"]
    )

    # ----------------------------------------------
    # Store results
    # ----------------------------------------------

    results.append({
        "stock": stock,
        "MAE": mae,
        "RMSE": rmse,
        "Mean Error": mean_error,
        "Mean Actual Volatility": mean_actual,
        "Mean Predicted Volatility": mean_predicted,
        "Correlation": correlation
    })


# ==================================================
# PART 5: SAVE STATISTICAL RESULTS
# ==================================================

statistics = pd.DataFrame(results)

statistics.to_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model1_statistics.csv",
    index=False
)


# ==================================================
# PART 6: PRINT RESULTS
# ==================================================

print("\n")
print("=" * 70)
print("MODEL 1 — NAÏVE VOLATILITY FORECAST")
print("=" * 70)

print(
    statistics.to_string(
        index=False
    )
)

print("\nStatistics saved to:")
print("C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model1_statistics.csv")