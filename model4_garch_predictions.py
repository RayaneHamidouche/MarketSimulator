import pandas as pd
import numpy as np
import os

from arch import arch_model


# --------------------------------------------------
# Setting up variables
# --------------------------------------------------

stocks = ["AAPL", "MSFT", "AMZN", "NVDA", "GOOGL"]

VOLATILITY_WINDOW = 20
TRAIN_SIZE = 0.8


# Create folder for predictions
os.makedirs("Predictions", exist_ok=True)


# Store all predictions
all_predictions = []


# --------------------------------------------------
# Processing each stock
# --------------------------------------------------

for stock in stocks:

    print(f"Processing {stock}...")

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    df = pd.read_csv(
        f"C:/PATH/MarketData/{stock}.csv",
        index_col=0
    )

    # Convert dates
    df.index = pd.to_datetime(
        df.index,
        dayfirst=True
    )

    # Sort chronologically
    df = df.sort_index()

    # Make sure close is numerical
    df["close"] = pd.to_numeric(df["close"])


    # --------------------------------------------------
    # Calculate the returns
    # --------------------------------------------------

    df["return"] = df["close"].pct_change()


    # --------------------------------------------------
    # Calculate the target volatility
    #
    # This is the same 20-day volatility used by
    # Models 1-3, so the models are comparable.
    # --------------------------------------------------

    df["volatility"] = (
        df["return"]
        .rolling(VOLATILITY_WINDOW)
        .std()
    )


    # --------------------------------------------------
    # Remove the missing values
    # --------------------------------------------------

    df = df.dropna(
        subset=[
            "return",
            "volatility"
        ]
    )


    # --------------------------------------------------
    # Train / Test split
    #
    # We use the first 80% to fit the model and
    # the final 20% to test it.
    # --------------------------------------------------

    split_index = int(
        len(df) * TRAIN_SIZE
    )

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]


    # --------------------------------------------------
    # Fit GARCH(1,1)
    #
    # ARCH expects returns rather than volatility.
    # Multiplying by 100 makes the numerical scale
    # easier for the model to work with.
    # --------------------------------------------------

    train_returns = train["return"] * 100

    model = arch_model(
        train_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    fitted_model = model.fit(
        disp="off"
    )


    # --------------------------------------------------
    # Expanding window forecast
    #
    # For each test day, refit the model using only
    # information available up to that day.
    # --------------------------------------------------

    predictions = []

    historical_returns = (
        train["return"] * 100
    ).copy()

    for i in range(len(test)):

        current_date = test.index[i]

        # Fit using all information available
        # before the prediction date
        model = arch_model(
            historical_returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="normal"
        )

        fitted_model = model.fit(
            disp="off"
        )

        # Forecast one day ahead
        forecast = fitted_model.forecast(
            horizon=1
        )

        # Forecast variance
        variance_forecast = (
            forecast.variance.iloc[-1, 0]
        )

        # Convert variance into volatility
        predicted_volatility = (
            np.sqrt(variance_forecast) / 100
        )

        actual_volatility = (
            test.loc[
                current_date,
                "volatility"
            ]
        )

        absolute_error = abs(
            predicted_volatility
            - actual_volatility
        )

        predictions.append({
            "date": current_date,
            "stock": stock,
            "predicted_volatility":
                predicted_volatility,
            "actual_volatility":
                actual_volatility,
            "absolute_error":
                absolute_error
        })

        # Add today's return to the historical
        # information available for the next forecast
        historical_returns = pd.concat([
            historical_returns,
            pd.Series(
                [test.loc[current_date, "return"] * 100],
                index=[current_date]
            )
        ])


    # Add this stock's predictions to the master list
    all_predictions.extend(predictions)


# --------------------------------------------------
# Creating the final dataframe
# --------------------------------------------------

all_predictions = pd.DataFrame(
    all_predictions
)


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

all_predictions.to_csv(
    "C:/PATH/Predictions/model4_garch_predictions.csv",
    index=False
)


# --------------------------------------------------
# Calculate the MAE
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
# Print results
# --------------------------------------------------

print("\nModel 4 - GARCH Mean Absolute Error:")
print(mae_results)


# --------------------------------------------------
# Save MAE
# --------------------------------------------------

mae_results.to_csv(
    "C:/PATH/Predictions/model4_mae.csv",
    index=False
)


print("\nFinished!")

print(
    "Predictions saved to:"
)

print(
    "C:/PATH/Predictions/model4_garch_predictions.csv"
)

print("\nMAE saved to:")

print(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model4_mae.csv"
)
