pip install pandas
pip install matplotlib
pip install streamlit
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
st.title("Time Series Analyzer")

uploaded_file = st.file_uploader("Upload your time series data (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
     
        st.error(f"Error loading data: {e}")
        st.stop()

    st.subheader("Uploaded Data")
    st.dataframe(df.head())
else:
    st.info("Please upload a time series data file to proceed.")
    st.stop()
date_column = st.selectbox("Select the date column:", df.columns)
value_column = st.selectbox("Select the value column:", [col for col in df.columns if col != date_column])
try:
    df[date_column] = pd.to_datetime(df[date_column])
except Exception as e:
    st.error(f"Error converting date column: {e}. Please ensure it's in a recognizable date format.")
    st.stop()
df.set_index(date_column, inplace=True)
df = df[[value_column]].sort_index()

#Obtaining the Line Diagram
st.subheader("Time Series Plot")
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[value_column])
plt.xlabel("Time")
plt.ylabel(value_column)
plt.title("Time Series Data")
st.pyplot(plt)

#Obtaining the decomposition plot
st.subheader("Decomposition Plot")
model=st.selectbox("Select the type of decomposition:",["Additive","Multiplicative"])
seasons=st.number_input("Enter number of seasons per year in the data", step=1,value=1)
from statsmodels.tsa.seasonal import seasonal_decompose
ts = df[value_column]
decomposition = seasonal_decompose(ts, model=model, period=seasons)
d_plot=decomposition.plot()
st.pyplot(d_plot)

#Obtaining ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
st.subheader("ACF Plot")
fig_acf=plot_acf(df[value_column])
st.pyplot(fig_acf)
st.subheader("PACF Plot")
fig_pacf=plot_pacf(df[value_column])
st.pyplot(fig_pacf)

#ADF Test
st.subheader("ADF Test")
st.write("Null Hypothesis: The time series is non-stationary")
st.write("Alternative Hypothesis: The time is stationary")
from statsmodels.tsa.stattools import adfuller
adf=adfuller(df[value_column])
t=adf[0]
p=adf[1]
st.write("The value of the test statistic is",t)
st.write("The p-value of the test is",p)
if p<0.05:
    st.write("Since the p value is less than the level of significance at 5% level of significance, we reject the null hypothesis")
    st.write("So, the time series is STATIONARY")
else:
    st.write("Since the p value is greater than the level of significance at 5% level of significance, we accept the null hypothesis")
    st.write("So, the time series is NON-STATIONARY")

#AutoARIMA Forecast
import pmdarima as pm
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

st.subheader("AutoARIMA Forecast")
n_periods = st.number_input("Select number of periods to forecast:", min_value=1, value=12, step=1)
with st.spinner("Fitting AutoARIMA model..."):
    model = pm.auto_arima(df[value_column], seasonal=True, m=seasons,
                          stepwise=True, suppress_warnings=True, error_action='ignore')
    st.success("AutoARIMA model fitted!")

# Display model summary
st.text("Model Summary:")
st.text(model.summary())

# Forecast future periods
forecast = model.predict(n_periods=n_periods)

# Create the forecast index based on the inferred frequency
last_date = df.index[-1]
inferred_freq = pd.infer_freq(df.index)
forecast_index = pd.date_range(start=last_date, periods=n_periods + 1, freq=inferred_freq)[1:]

# Combine forecast into a dataframe
forecast_df = pd.DataFrame({value_column: forecast}, index=forecast_index)

# Plot forecast
st.subheader("Forecast Plot")
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[value_column], label='Historical')
plt.plot(forecast_df.index, forecast_df[value_column], label='Forecast', linestyle='--')
plt.xlabel("Time")
plt.ylabel(value_column)
plt.title("Time Series Forecast")
plt.legend()
st.pyplot(plt)

# Show forecast values
st.subheader("Forecasted Values")
st.dataframe(forecast_df)

