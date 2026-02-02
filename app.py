import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



model = load_model(
    r"C:\Users\Medhansh Mathur\OneDrive\Desktop\stock_predictor\Stock Predictions Model.keras"
)


st.header("📈 Stock Price Prediction App")


stock = st.text_input("Enter Stock Ticker")

if not stock:
    st.info("Enter a stock symbol to continue")
    st.stop()

stock = stock.upper()

start = "2012-01-01"
end = "2025-12-31"

data = yf.download(stock, start, end)

if data.empty:
    st.error("Invalid stock symbol")
    st.stop()


st.subheader("Stock Data")
st.write(data)


# ---------------- DATA PREP ----------------
data_train = pd.DataFrame(data.Close[:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)


# ---------------- PLOTS ----------------
st.subheader("Price vs MA50")
ma_50 = data.Close.rolling(50).mean()

fig1 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label="Original Price")
plt.plot(ma_50, label="MA 50")
plt.legend()
st.pyplot(fig1)


st.subheader("Price vs MA50 vs MA100")
ma_100 = data.Close.rolling(100).mean()

fig2 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label="Original Price")
plt.plot(ma_50, label="MA 50")
plt.plot(ma_100, label="MA 100")
plt.legend()
st.pyplot(fig2)


st.subheader("Price vs MA100 vs MA200")
ma_200 = data.Close.rolling(200).mean()

fig3 = plt.figure(figsize=(8, 6))
plt.plot(data.Close, label="Original Price")
plt.plot(ma_100, label="MA 100")
plt.plot(ma_200, label="MA 200")
plt.legend()
st.pyplot(fig3)



x, y = [], []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predicted = model.predict(x)

scale = 1 / scaler.scale_
predicted = predicted * scale
y = y * scale


st.subheader("Original Price vs Predicted Price")

fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, label="Original Price")
plt.plot(predicted, label="Predicted Price")
plt.legend()
st.pyplot(fig4)
