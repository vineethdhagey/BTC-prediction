# BTC-prediction
Predicts the price of BTC in USD based on historical data and senitment analysis

1) Developed a Bitcoin price prediction model using LSTM and integrating FinBERT-based sentiment analysis in python on real-time Reddit data to forecast hourly price movements. Utilized key libraries such as TensorFlow, Transformers (Hugging Face), PRAW, Pandas, yFinance, and Scikit-learn.

2) Created a data pipeline that merges hourly Bitcoin price data from Yahoo Finance with Reddit sentiment scores, making sure the timestamps line up for accurate modeling.

3) Created a user-friendly web interface using streamlit, where users can choose how far into the future they want to predict Bitcoin prices and see clear, easy-to-understand visuals showing the expected price trend and percentage change


## 📁 Project Structure

```

BTC-Prediction/
│
├── app.py # Main Streamlit app
├── btc_prediction.ipynb # Python with ML
├── lstm_model.h5 #saved LSTM model
├── model_backend.py #backend preprocessing code
├── results.png
├── train_model.py #model training code
└── README.md # Project documentation

```

---


### ⚙️ Installation & Setup
**1) Clone the repository:**
 
 ```bash
 git clone https://github.com/dvinzzzz/BTC-prediction.git
 cd BTC-prediction
```

**2) Create and activate a virtual environment**

   ```bash
    Windows:
   python -m venv venv
  venv\Scripts\activate
   ```
**3) Install dependencies**

   ```bash
   pip install -r requirements.txt

```




**4) Run the app:**
 ```bash
streamlit run app.py

```




<img width="940" height="560" alt="output2" src="https://github.com/user-attachments/assets/cc588256-52e3-4c7a-a0a7-f27dafa677f6" />

<img width="994" height="366" alt="results" src="https://github.com/user-attachments/assets/e53c787f-b6a2-4b27-8c95-f68d651cd346" />
