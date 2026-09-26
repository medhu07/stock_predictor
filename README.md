# stock_predictor
**Stock Price Prediction App (Streamlit + ML)**

This project is a Stock Price Prediction Web App built using Python, Streamlit, and a trained deep learning model (Keras/TensorFlow).
It fetches stock market data using Yahoo Finance and visualizes trends along with model predictions.

**Tech Stack**
1)Python 3.9+
2)Streamlit
3)NumPy
4)Pandas
5)Matplotlib
6)yfinance
7)Scikit-learn
8)TensorFlow
9)Keras

**MAKE SURE LIBRARIES SHOWN ABOVE ARE INSTALLED IN YOUR JUPYTER NOTEBOOK**


1) after running the code in your notebook, copy the path of the saved file after training , and enter the same path while loading the model in the web app code

2) after running the python code, use this -> [ streamlit run app.py ] as it is in the terminal of the python output to open the webpage
   

# 📈 Stock Price Prediction using LSTM

An end to end stock price prediction project using **Long Short Term Memory (LSTM)** neural networks.  
The model is trained on historical stock market data and the predictions are presented through an interactive **Streamlit web application**.

## 🚀 Live Demo

👉 **[Try the Live App](https://stockpredictor-ccfv78hhr9dc66hmdyaosn.streamlit.app/)**

## 📌 Project Overview

Stock prices are time series data where past price movements can be used to identify patterns and trends.

In this project, an **LSTM based deep learning model** is trained on historical stock data and used to predict stock prices.

The project covers the complete workflow:

* Fetching historical stock market data
* Data preprocessing and normalization
* Creating sequences for time series prediction
* Training an LSTM neural network
* Evaluating predictions against actual prices
* Visualizing stock price trends
* Deploying the model using Streamlit

## 🧠 Model Architecture

The project uses a stacked LSTM architecture:

```text
Input Sequence
      ↓
LSTM (50 units)
      ↓
Dropout (20%)
      ↓
LSTM (60 units)
      ↓
Dropout (30%)
      ↓
LSTM (80 units)
      ↓
Dropout (40%)
      ↓
LSTM (120 units)
      ↓
Dropout (50%)
      ↓
Dense (1)
      ↓
Predicted Stock Price



