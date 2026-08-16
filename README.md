# MarketSimulator

# Equity Volatility Forecasting

A Python project investigating how different statistical models can forecast short-term equity volatility.

## Overview

The project uses daily market data from five US equities:

* Apple (AAPL)
* Amazon (AMZN)
* Alphabet (GOOGL)
* Microsoft (MSFT)
* Nvidia (NVDA)

Daily returns are used to calculate 20-day rolling volatility. Four forecasting models are then compared:

1. **Naïve model** — uses the latest observed volatility as the next-day forecast.
2. **Moving average model** — uses the average of recent volatility observations.
3. **Linear regression** — uses recent volatility, absolute returns and relative trading volume.
4. **GARCH(1,1)** — models changes in volatility using past returns and volatility.

The models are evaluated on an out-of-sample test period using **Mean Absolute Error (MAE)**.

## Project Structure

```text
├── MarketData/
│   └── Historical market data
│
├── models/
│   ├── model1_naive.py
│   ├── model2_moving_average.py
│   ├── model3_regression.py
│   └── model4_garch.py
│
├── Predictions/
│   └── Model predictions and statistical results
│
├── analysis/
│   └── Model comparison and visualisation
│
├── report/
│   └── Project report
│
└── README.md
```

## Results

The naïve model achieved the lowest MAE for four of the five assets, while the regression model performed best for NVDA. The GARCH model performed weakest under the conditions of this study.

The results suggest that greater model complexity does not necessarily lead to more accurate volatility forecasts, particularly when the target volatility measure is highly persistent.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* ARCH

## Running the Project

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn arch
```

The individual model files can then be run to generate forecasts and statistical results.

## Report

An outline of the methodology and results is provided in the project report.
