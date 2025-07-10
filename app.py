import streamlit as st
from model_backend import run_prediction
from PIL import Image

st.set_page_config(page_title="DSS for BTC Prediction", layout="centered")

logo = Image.open("assets/logo.png")
st.image(logo, width=120)

st.title("💡 Decision Support System for BTC-USD Price Prediction")

# Input: prediction horizon (in hours)
hours = st.number_input("Enter number of hours to predict ahead:", min_value=1, max_value=168, value=24)

if st.button("Predict"):
    with st.spinner("Running prediction..."):
        current, predicted = run_prediction(hours)
        change = predicted - current
        percent_change = (change / current) * 100

        # Show price values
        st.subheader(f"📊 Current Price: ${current:,.2f}")
        

        # Direction logic
        if change > 0:
            arrow = Image.open("assets/up.png")
            st.success(f"Price is predicted to increase 📈 (+${abs(change):,.2f})")
        else:
            arrow = Image.open("assets/down.png")
            st.error(f"Price is predicted to decrease 📉 (-${abs(change):,.2f})")

        st.image(arrow, width=50)

        # Final Forecast Summary
        direction = "increasing" if change > 0 else "decreasing"
        emoji = "✅ Yes" if change > 0 else "❌ No"
        arrow_emoji = "📈" if change > 0 else "📉"

        st.markdown("### 🔮 Final Forecast Summary:")
        st.markdown(f"{emoji}, the price is *{direction}* by *{abs(percent_change):.2f}%* in the next *{hours} hours*. {arrow_emoji}")

        