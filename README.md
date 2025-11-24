# BTC-Prediction: Bitcoin Price Forecasting Using Sentiment Analysis

A Decision Support System (DSS) designed to predict Bitcoin (BTC) prices in USD by leveraging historical price data along with real-time sentiment analysis gathered from Reddit. This system employs Long Short-Term Memory (LSTM) neural networks integrated with FinBERT-based sentiment scoring to provide hourly price forecasts.

## 🚀 Key Features

- **Real-Time Price Forecasting:** Predict BTC prices up to 168 hours (7 days) ahead by analyzing past price trends and sentiment signals.
- **Sentiment Integration:** Utilizes sentiment scores derived from Reddit discussions in cryptocurrency-related subreddits through the FinBERT model.
- **Interactive Web Application:** User-friendly Streamlit interface for model predictions and visualizations.
- **Insightful Visualizations:** Displays current price, predicted price variations, percentage changes, and directional arrows indicating market trends.
- **Robust Data Pipeline:** Merges hourly BTC price data from Yahoo Finance with corresponding sentiment data, aligned accurately on timestamps.
- **Model Evaluation:** R²=0.85, 85% of the predictions made by my model follows the actual trend.

## 🛠 Technology Stack

- **Machine Learning:** TensorFlow, Scikit-learn
- **Natural Language Processing:** Hugging Face Transformers, FinBERT
- **Data Manipulation:** Pandas, NumPy, yFinance
- **Web Scraping:** PRAW (Python Reddit API Wrapper)
- **Visualization:** Streamlit, Pillow
- **Utilities:** Joblib, SciPy

## 📁 Project Structure

```
BTC-Prediction/
│
├── app.py                 # Streamlit app for user interaction
├── model_backend.py       # Backend handling model inference and predictions
├── train_model.py         # Script to prepare data and train the LSTM model
├── lstm_model.h5          # Saved trained LSTM model
├── scaler.save            # Scaler object for data normalization
├── requirements.txt       # Project dependencies
├── assets/                # Static assets such as images and icons
│   ├── logo.png
│   ├── up.png
│   ├── down.png
│   └── neutral.png
└── README.md              # Documentation and usage guide
```


                                                                       
                                                                       
## Architecture Diagram                                                                
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       
                                                                        ┌──────────────────────────┐
                                                                        │    Data Collection       │
                                                                        │ - BTC price (Yahoo)      │
                                                                        │ - Reddit posts (PRAW)    │
                                                                        └──────────┬───────────────┘
                                                                                   ▼
                                                                        ┌──────────────────────────┐
                                                                        │    Sentiment Processing  │
                                                                        │ - FinBERT sentiment      │
                                                                        │ - Pos / Neu / Neg scores │
                                                                        └──────────┬───────────────┘
                                                                                   ▼
                                                                        ┌──────────────────────────┐
                                                                        │    Data Merging + Prep   │
                                                                        │ - Align timestamps       │
                                                                        │ - Merge price + sentiment│
                                                                        │ - Scaling + sequences    │
                                                                        └──────────┬───────────────┘
                                                                                   ▼
                                                                        ┌──────────────────────────┐
                                                                        │    Model Training        │
                                                                        │ - LSTM sequence model    │
                                                                        │ - Save model + scaler    │
                                                                        └──────────┬───────────────┘
                                                                                   ▼
                                                                        ┌──────────────────────────┐
                                                                        │    Prediction            │
                                                                        │ - Forecast 1–168 hrs     │
                                                                        │ - Price + % change       │
                                                                        │ - Direction indicator    │
                                                                        └──────────┬───────────────┘
                                                                                   ▼
                                                                        ┌──────────────────────────┐
                                                                        │    Visualization         │
                                                                        │ - Streamlit interface    │
                                                                        │ - Results + icons        │
                                                                        │ - User interaction       │
                                                                        └──────────────────────────┘





















## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9 or newer
- Git (optional for cloning)
- Internet access for downloading dependencies and datasets

### Installation Steps

1. **Clone the repository (if applicable):**
   ```bash
   git clone https://github.com/vineethdhagey/BTC-prediction.git
   cd BTC-prediction
   ```

2. **Set up a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

   The application will open at `http://localhost:8501` in your default web browser.

## 📖 Usage Guidelines

1. Open the application in a browser.
2. Specify the forecast horizon by entering the number of hours ahead (1 to 168).
3. Click the **Predict** button to generate BTC price predictions.
4. Review the current BTC price, projected price change, percentage variation, and visual trend indicators.

## 🔍 System Workflow

1. **Data Acquisition:**
   - Retrieves hourly BTC price data from Yahoo Finance covering the past 30 days.
   - Gathers Reddit posts and comments from relevant cryptocurrency subreddits via Reddit API.

2. **Sentiment Evaluation:**
   - Applies the FinBERT financial sentiment transformer to quantify sentiment in Reddit content.
   - Aggregates sentiment scores to an hourly granularity matching price data timestamps.

3. **Data Preparation:**
   - Merges and synchronizes price and sentiment datasets by hourly intervals.
   - Normalizes features through MinMaxScaler.
   - Constructs input sequences with a window size of 90 hours (~3.75 days) for model input.

4. **Model Training:**
   - Trains an LSTM network on combined price and sentiment data.
   - Employs early stopping for training optimization and to avoid overfitting.

5. **Prediction & Visualization:**
   - Generates forecasts for up to one week ahead (168 hours).
   - Presents predictions interactively through Streamlit with accompanying visual cues.

## 📸 Visual Examples

### Application Interface
<img width="940" height="560" alt="output2" src="https://github.com/user-attachments/assets/bbc9ff4e-6286-4284-9bfc-f531dcccef4f" />



### Forecast Results

<img width="994" height="366" alt="results" src="https://github.com/user-attachments/assets/a56afe89-0831-403e-ac92-64a23b4521f7" />



## 📞 Contact

This project was developed by : **Vineeth Dhagey** and **Ekshith Satnur** as a part of our coursework **"Decision Support System"**

Special Thanks to our **professor** who gave us the right guidance for this!






## 📜 License

This project is licensed under the MIT License.

---

For inquiries or contributions, please open an issue or contact the maintainer.

