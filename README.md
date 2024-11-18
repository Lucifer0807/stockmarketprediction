# Stock Price Prediction - S&P 500 Model

This project is a machine learning-based model that predicts whether the closing price of the S&P 500 index will go up or not tomorrow. The model uses a **Random Forest Classifier** to make the predictions based on various features such as open, high, low, volume, and close prices.

## Features
- **Model Type:** Random Forest Classifier
- **Data Source:** Yahoo Finance (`yfinance` package)
- **Prediction Target:** Predicts whether the S&P 500 closing price will increase the next day (binary classification: 1 = increase, 0 = decrease)
- **Backtesting:** A backtesting function that evaluates the model's performance over time
- **Rolling Averages & Trend:** Incorporates rolling averages and trend features to improve prediction accuracy

## Project Setup

### Prerequisites
Make sure you have Python 3.x installed and use the `requirements.txt` file to install the necessary dependencies:

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd <your-project-folder>

2. Install required libraries:
  ```bash
  pip install -r requirements.txt
  ```

###Running the Application
Once the dependencies are installed, you can launch the Streamlit application to interact with the model.

1. Run the following command in your terminal:

  ```bash
  streamlit run app.py
  ```
2. This will open a new tab in your web browser with the Streamlit interface where you can visualize the model's predictions and explore the data.

### How It Works
- Data Collection: The model uses historical S&P 500 data, which is fetched using the yfinance library.
- Model Training: A Random Forest Classifier is trained on historical data to predict whether the S&P 500 closing price will go up the next day.
- Prediction: The app predicts tomorrow's movement (up or down) based on the latest data.

### Backtesting
- The backtesting function evaluates the model by splitting the historical data into training and test sets. It checks the prediction accuracy over multiple iterations using the Random Forest Classifier.
- The app also supports generating predictions with various horizon periods (e.g., 2, 5, 60, 250 days) using rolling averages.

### Model Performance
Accuracy Metrics: Precision score is used to evaluate the model’s performance. The app displays prediction accuracy over different time horizons.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgements
- yfinance for fetching historical stock data
- Streamlit for creating interactive web applications
- Scikit-learn for implementing machine learning models
