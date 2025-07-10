import praw
import pandas as pd
import re
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Collect Reddit Data
reddit = praw.Reddit(
    client_id='yZgp5fHdkhZQwGSCQ6Of4Q',
    client_secret='yZgmiMH34SQlf2efwsf1zIeqWXEvoQ',
    user_agent='Bitcoin Sentiment Analysis'
)

subreddits = ['Bitcoin', 'CryptoCurrency', 'BitcoinMarkets', 'btc']
posts = []

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    for submission in subreddit.search('Bitcoin', limit=50):
        try:
            submission.comments.replace_more(limit=0)
            top_comments = [comment.body for comment in submission.comments.list()[:20]]
            comment_text = " ".join([clean_text(comment) for comment in top_comments])
            title_clean = clean_text(submission.title)
            selftext_clean = clean_text(submission.selftext)
            combined_text = f"{title_clean} {selftext_clean} {comment_text}"
            posts.append([
                title_clean,
                selftext_clean,
                comment_text,
                combined_text,
                datetime.utcfromtimestamp(submission.created_utc),
                subreddit_name
            ])
        except Exception as e:
            print(f"⚠️ Skipping a post in r/{subreddit_name} due to error: {e}")
            continue

df_reddit = pd.DataFrame(posts, columns=['title', 'selftext', 'comments', 'content', 'created_utc', 'subreddit'])

# Step 2: Analyze Sentiment
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def get_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = softmax(logits.numpy()[0])
    sentiment = scores[2] - scores[0]
    return sentiment

df_reddit['content'] = df_reddit['title'] + ' ' + df_reddit['selftext'] + ' ' + df_reddit['comments']
df_reddit['sentiment'] = df_reddit['content'].apply(get_finbert_sentiment)

# Step 3: Get BTC price data
# Step 3: Get BTC price data
# Define time range
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

# Download BTC data
btc_data = yf.download('BTC-USD', start=start_date, end=end_date, interval='1h')

# Fix multilevel columns
if isinstance(btc_data.columns, pd.MultiIndex):
    btc_data.columns = btc_data.columns.get_level_values(0)  # Keep just the first level

# Reset index
btc_data.reset_index(inplace=True)

# Rename 'index' or confirm 'Datetime' exists
btc_data.rename(columns={'index': 'Datetime'}, inplace=True)

# Confirm column structure
print("btc_data.columns:", btc_data.columns)

# Remove timezone from Reddit timestamps
df_reddit['created_utc'] = pd.to_datetime(df_reddit['created_utc']).dt.tz_localize(None)
btc_data['Datetime'] = pd.to_datetime(btc_data['Datetime']).dt.tz_localize(None)

# Create hourly timestamps
df_reddit['hour'] = df_reddit['created_utc'].dt.floor('H')
btc_data['hour'] = btc_data['Datetime'].dt.floor('H')

# Group Reddit sentiment by hour
sentiment_hourly = df_reddit.groupby('hour', as_index=False)['sentiment'].mean()

# ✅ Ensure btc_data has no MultiIndex
btc_data.columns = [col if isinstance(col, str) else col[0] for col in btc_data.columns]

# ✅ Merge on the hour column
data = pd.merge(btc_data, sentiment_hourly, on='hour', how='left')

# Fill missing sentiment with neutral
data['sentiment'].fillna(0, inplace=True)

# Select required columns
data = data[['Datetime', 'Close', 'sentiment']]

# Normalize
from sklearn.preprocessing import MinMaxScaler
import joblib
scaler = MinMaxScaler()
data[['Close', 'sentiment']] = scaler.fit_transform(data[['Close', 'sentiment']])
joblib.dump(scaler, 'scaler.save')


# Step 6: Create sequences
def create_sequences(data, time_steps=90):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

dataset = data[['Close', 'sentiment']].values
time_steps = 90
X, y = create_sequences(dataset,time_steps)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 7: Build and train model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.1),
    LSTM(128),
    Dropout(0.1),
    Dense(1)
])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Step 8: Save model
model.save("lstm_model.h5")
print("✅ Model trained and saved as lstm_model.h5")
