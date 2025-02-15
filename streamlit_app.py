




import streamlit as st 
import pandas as pd 
import yfinance as yf 
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

#streamlit Ui
st.set_page_config(page_title= "Time series forecasting", layout="wide")
st.title("Time series forecasting with prophet")


#sidebar options
st.sidebar.header("User Input")
ticker = st.sidebar.text_input("Enter Stock Ticker", "NVDA")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-02-02"))
forcast_horizon = st.sidebar.slider("Select Forcast Horizon (days) ",30,90)

if st.sidebar.button("Fetch Data"):
        #fetch data
        data = yf.download(ticker, start = start_date, end = end_date)[['Close']].reset_index()
        data.columns = ['ds','y']

        #display data preview
        st.subheader(f"data for {ticker}")
        st.dataframe(data.tail())

        #Prophet Model
        model = Prophet(daily_seasonality = False, yearly_seasonality= True, Weekly_seasonality = True)
        model.fit(data)

        # Forcast
        future = model.make_future_dataframe(periods= forcast_horizon)
        forecast = model.predict(future)

        # plot results using plotly
        fig = plot_plotly(model,forcast)
        st.subheader("Forcasted Stock Price")
        st.plotly_chart(fig)


        # Additional Trend Components
        st.subheader("Trend and Seasonal Components")
        st.write("Yearly and Weekly Trends")
        st.write(model.plot_components(forcast))
