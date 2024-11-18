import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import streamlit as st

# Fetching the data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

# Preprocessing
sp500["tomorrow"] = sp500["Close"].shift(-1)
sp500["Target"] = (sp500["tomorrow"] > sp500["Close"]).astype(int)

# Filter data for the years starting from 1990
sp500 = sp500.loc["1990-01-01":].copy()

# Training and Testing split
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

# Define predictors and train the model
predictors = ["Close", "Volume", "Open", "High", "Low"]
rf = RandomForestClassifier(random_state=1)
rf.fit(train[predictors], train["Target"])

# Model prediction
preds = rf.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Evaluation - precision score
precision = precision_score(test["Target"], preds)

# Function to predict and evaluate
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Backtesting with the model
predictions = backtest(sp500, rf, predictors)

# Display precision score
backtest_precision = precision_score(predictions["Target"], predictions["Predictions"])

# Rolling averages and trend calculation
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]

sp500 = sp500.dropna()

# Re-train model with new predictors
rf = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Prediction with probability thresholds
def predict_with_threshold(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.6] = 1
    preds[preds < 0.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Run backtest with new predictors and threshold model
predictions = backtest(sp500, rf, new_predictors)

# Final precision score
final_precision = precision_score(predictions["Target"], predictions["Predictions"])

# Streamlit UI components
st.title("S&P 500 Price Prediction")
st.write("This model predicts whether the S&P 500 price will go up or not tomorrow based on past data.")

# Display the data plot
st.write("### S&P 500 Closing Price")
st.line_chart(sp500["Close"])

# Display results
st.write(f"**Backtest Precision Score:** {backtest_precision:.2f}")
st.write(f"**Final Model Precision Score (with threshold):** {final_precision:.2f}")

# Show latest prediction
st.write("### Latest Prediction")
st.write(f"Prediction for tomorrow: {'Price will go up' if preds[-1] == 1 else 'Price will not go up'}")
