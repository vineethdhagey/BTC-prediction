# BTC-Prediction: Bitcoin Price Forecasting with Sentiment Analysis

A Decision Support System (DSS) for predicting Bitcoin (BTC) prices in USD using historical data and real-time sentiment analysis from Reddit. This project combines Long Short-Term Memory (LSTM) neural networks with FinBERT-based sentiment scoring to forecast hourly price movements.

## 🚀 Key Features

- **Real-time Price Prediction**: Forecast BTC prices for up to 168 hours ahead based on historical trends and sentiment.
- **Sentiment Integration**: Analyzes Reddit discussions from subreddits like r/Bitcoin and r/CryptoCurrency using FinBERT for sentiment scoring.
- **User-Friendly Interface**: Interactive Streamlit web app for easy prediction input and visualization.
- **Visual Insights**: Displays current price, predicted changes, percentage variations, and directional indicators (up/down arrows).
- **Data Pipeline**: Merges hourly BTC price data from Yahoo Finance with sentiment scores, ensuring timestamp alignment.

## 🛠 Technology Stack

- **Machine Learning**: TensorFlow, Scikit-learn
- **NLP & Sentiment**: Transformers (Hugging Face), FinBERT
- **Data Handling**: Pandas, NumPy, yFinance
- **Web Scraping**: PRAW (Reddit API)
- **Visualization**: Streamlit, Pillow
- **Utilities**: Joblib, SciPy

## 📁 Project Structure

```
BTC-Prediction/
│
├── app.py                 # Main Streamlit application
├── model_backend.py       # Prediction backend logic
├── train_model.py         # Model training script
├── lstm_model.h5          # Trained LSTM model
├── scaler.save            # Data scaler for normalization
├── requirements.txt       # Python dependencies
├── assets/                # Images and logos
│   ├── logo.png
│   ├── up.png
│   ├── down.png
│   └── neutral.png
└── README.md              # Project documentation
```

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- Git

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vineethdhagey/BTC-prediction.git
   cd BTC-prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

The app will open in your default browser at `http://localhost:8501`.

## 📖 Usage

1. Open the Streamlit app in your browser.
2. Enter the number of hours you want to predict ahead (1-168 hours).
3. Click the "Predict" button.
4. View the current BTC price, predicted price change, percentage change, and directional indicators.

## 📸 Screenshots

### Main Interface
![Main Interface](https://github.com/user-attachments/assets/cc588256-52e3-4c7a-a0a7-f27dafa677f6)

### Prediction Results
![Prediction Results](https://github.com/user-attachments/assets/e53c787f-b6a2-4b27-8c95-f68d651cd346)

## 🔍 How It Works

1. **Data Collection**: Fetches hourly BTC price data from Yahoo Finance and Reddit posts from relevant subreddits.
2. **Sentiment Analysis**: Uses FinBERT to score sentiment from Reddit content (positive, neutral, negative).
3. **Data Merging**: Aligns price and sentiment data by hourly timestamps.
4. **Model Training**: Trains an LSTM model on sequences of price and sentiment data.
5. **Prediction**: The trained model forecasts future prices based on recent data and sentiment trends.
6. **Visualization**: Streamlit app displays predictions with intuitive visuals and summaries.

This project demonstrates the integration of financial data with social sentiment for improved price forecasting accuracy.
