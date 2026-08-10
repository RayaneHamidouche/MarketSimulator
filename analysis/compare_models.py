import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Load the MAE files
# --------------------------------------------------

model1 = pd.read_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model1_mae.csv"
)

model2 = pd.read_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model2_mae.csv"
)

model3 = pd.read_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model3_mae.csv"
)

model4 = pd.read_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model4_mae.csv"
)


# --------------------------------------------------
# Rename the columns
# --------------------------------------------------

model1 = model1.rename(
    columns={"MAE": "Model 1"}
)

model2 = model2.rename(
    columns={"MAE": "Model 2"}
)

model3 = model3.rename(
    columns={"MAE": "Model 3"}
)

model4 = model4.rename(
    columns={"MAE": "Model 4"}
)


# --------------------------------------------------
# Combining the results
# --------------------------------------------------

comparison = model1.merge(
    model2,
    on="stock"
)

comparison = comparison.merge(
    model3,
    on="stock"
)

comparison = comparison.merge(
    model4,
    on="stock"
)


# --------------------------------------------------
# Finding the best model for each stock
# --------------------------------------------------

model_columns = [
    "Model 1",
    "Model 2",
    "Model 3",
    "Model 4"
]

comparison["Best Model"] = (
    comparison[model_columns]
    .idxmin(axis=1)
)


# --------------------------------------------------
# Print Results
# --------------------------------------------------

print("\nMODEL COMPARISON")
print("----------------")

print(comparison.to_string(index=False))


# --------------------------------------------------
# The overall average MAE
# --------------------------------------------------

average_mae = (
    comparison[model_columns]
    .mean()
    .sort_values()
)

print("\nAverage MAE across stocks:")
print(average_mae)


# --------------------------------------------------
# Counting how many stocks each model won
# --------------------------------------------------

wins = (
    comparison["Best Model"]
    .value_counts()
)

print("\nBest model by number of stocks:")
print(wins)


# --------------------------------------------------
# Saving the comparison
# --------------------------------------------------

comparison.to_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model_comparison.csv",
    index=False
)

wins.to_csv(
    "C:/Users/heatw/OneDrive - Twyford Academies/Documents/Predictions/model_wins.csv"
)


# --------------------------------------------------
# Plotting the
# --------------------------------------------------

comparison_plot = comparison.set_index("stock")[
    model_columns
]

comparison_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Volatility Forecasting Model Comparison")
plt.xlabel("Stock")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.grid(axis="y")

plt.tight_layout()
plt.show()