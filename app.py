import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    with open("risk_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.set_page_config(page_title="Stock Risk Predictor", page_icon="📈", layout="wide")

st.title("📊 Stock Risk Prediction App")
st.markdown("""
Use this app to predict the **risk level** of a stock based on key market indicators.
Enter the details below and click **Predict Risk**.
""")

st.sidebar.header("Input Stock Features")

def user_input_features():
    Open = st.sidebar.number_input("Open", min_value=0.0, step=0.1)
    Previous_Close = st.sidebar.number_input("Previous Close", min_value=0.0, step=0.1)
    Volume = st.sidebar.number_input("Volume", min_value=0.0, step=1000.0)
    Value_Lacs = st.sidebar.number_input("Value (Lacs)", min_value=0.0, step=10.0)
    VWAP = st.sidebar.number_input("VWAP", min_value=0.0, step=0.1)
    Mkt_Cap = st.sidebar.number_input("Market Cap (Rs. Cr.)", min_value=0.0, step=1.0)
    High = st.sidebar.number_input("High", min_value=0.0, step=0.1)
    Low = st.sidebar.number_input("Low", min_value=0.0, step=0.1)
    UC_Limit = st.sidebar.number_input("UC Limit", min_value=0.0, step=0.1)
    LC_Limit = st.sidebar.number_input("LC Limit", min_value=0.0, step=0.1)
    W52_High = st.sidebar.number_input("52W High", min_value=0.0, step=0.1)
    W52_Low = st.sidebar.number_input("52W Low", min_value=0.0, step=0.1)
    All_Time_High = st.sidebar.number_input("All Time High", min_value=0.0, step=0.1)
    Avg_20D_Volume = st.sidebar.number_input("20D Avg Volume", min_value=0.0, step=1000.0)
    Book_Value = st.sidebar.number_input("Book Value Per Share", min_value=0.0, step=0.1)

    data = {
        'Open': Open,
        'Previous Close': Previous_Close,
        'Volume': Volume,
        'Value (Lacs)': Value_Lacs,
        'VWAP': VWAP,
        'Mkt Cap (Rs. Cr.)': Mkt_Cap,
        'High': High,
        'Low': Low,
        'UC Limit': UC_Limit,
        'LC Limit': LC_Limit,
        '52W High': W52_High,
        '52W Low': W52_Low,
        'All Time High': All_Time_High,
        '20D Avg Volume': Avg_20D_Volume,
        'Book Value Per Share': Book_Value
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

if st.button("🔍 Predict Risk"):
    try:
        prediction = model.predict(input_df)[0]
        st.subheader("🧭 Predicted Risk Level:")
        st.success(prediction)

        if prediction == "Low":
            st.info("💼 Suggested Stocks: HDFC Bank, Infosys, TCS")
        elif prediction == "Medium":
            st.warning("📈 Suggested Stocks: Tata Motors, ICICI Bank, Larsen & Toubro")
        else:
            st.error("🔥 Suggested Stocks: Zomato, Paytm, Adani Power")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Created with ❤️ by Krati Nawal")
st.sidebar.write("[View on GitHub](https://github.com/Krati26/ML-Model)")
