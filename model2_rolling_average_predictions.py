import pandas as pd
import os


# --------------------------------------------------
# Setting up variables
# --------------------------------------------------

stocks = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

VOLATILITY_WINDOW = 20
FORECAST_WINDOW = 5


# Create folder for predictions
os.makedirs("Predictions", exist_ok=True)


# Store all prediction rows
all_predictions = []


# --------------------------------------------------
# Processing each stock
# --------------------------------------------------

for stock in stocks:

    print(f"Processing {stock}...")

    # Load data
    df = pd.read_csv(
        f"C:/PATH/MarketData/{stock}.csv",
        index_col=0
    )

    # Convert dates
    df.index = pd.to_datetime(
        df.index,
        dayfirst=True
    )

    # Make sure data is chronological
    df = df.sort_index()

    # Make sure close is numerical
    df["close"] = pd.to_numeric(df["close"])

    # --------------------------------------------------
    # Calculate daily returns
    # --------------------------------------------------

    df["return"] = df["close"].pct_change()

    # --------------------------------------------------
    # Calculate the 20 day volatility
    # --------------------------------------------------

    df["volatility"] = (
        df["return"]
        .rolling(VOLATILITY_WINDOW)
        .std()
    )

    # --------------------------------------------------
    # Model 2
    #
    # Predicts tomorrow's volatility using the
    # average volatility from the previous 5 days.
    # --------------------------------------------------

    df["predicted_volatility"] = (
        df["volatility"]
        .rolling(FORECAST_WINDOW)
        .mean()
        .shift(1)
    )

    # Remove rows where there isn't enough information
    df = df.dropna(
        subset=[
            "volatility",
            "predicted_volatility"
        ]
    )

    # --------------------------------------------------
    # Calculating the prediction error
    # --------------------------------------------------

    df["absolute_error"] = (
        df["predicted_volatility"]
        - df["volatility"]
    ).abs()

    # --------------------------------------------------
    # Storing the results
    # --------------------------------------------------

    for date, row in df.iterrows():

        all_predictions.append({
            "date": date,
            "stock": stock,
            "predicted_volatility":
                row["predicted_volatility"],
            "actual_volatility":
                row["volatility"],
            "absolute_error":
                row["absolute_error"]
        })


# --------------------------------------------------
# Creating a dataframe
# --------------------------------------------------

all_predictions = pd.DataFrame(all_predictions)


# --------------------------------------------------
# Saving the predctions to .csv file
# --------------------------------------------------

all_predictions.to_csv(
    "C:/PATH/Predictions/model2_moving_average_predictions.csv",
    index=False
)


# --------------------------------------------------
# Calculating the MAE for each stock
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


# Convert to DataFrame
mae_results = pd.DataFrame(mae_results)


# --------------------------------------------------
# Printing results
# --------------------------------------------------

print("\nModel 2 - Mean Absolute Error:")
print(mae_results)


# --------------------------------------------------
# Saving the Mean Absolute Error
# --------------------------------------------------

mae_results.to_csv(
    "C:/PATH/Predictions/model2_mae.csv",
    index=False
)


print("\nFinished!")

print(
    "Predictions saved to:"
)

print(
    "C:/PATH/Predictions/model2_mae.csv"
)

print("\nMAE saved to:")
print("C:/PATH/Predictions/model2_mae.csv")
