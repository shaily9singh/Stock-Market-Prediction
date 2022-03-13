from tkinter.ttk import LabelFrame
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px 
import seaborn as sns
import chart_studio.plotly as py
import streamlit as st
from keras.models import load_model
import pandas_datareader as data



st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start = st.date_input('Start', value = pd.to_datetime('2010-01-01'))
end = st.date_input('End', value= pd.to_datetime('today'))
data = data.DataReader(user_input, 'yahoo', start, end)
data = data.reset_index()
st.subheader('Descriptive Data from 2010 - 2021')
st.write(data.describe())


#Closing data view
fig = px.line(data, x =data.Date, y=data.Close, labels={'x': 'Date', 'y': 'Closing_Price'})

fig.update_layout(title='Closing Data with Dates 2010-2021',
                 xaxis_title='Date', yaxis_title='Price',
                 xaxis_rangeslider_visible=True)  

st.plotly_chart(fig) 


#Moving Average view
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.Date, y=data.Close,
                        mode='lines', name='Stock_Close'))
fig.add_trace(go.Scatter(x=data.Date, y=ma100,
                        mode='lines', name='Stock_Close with ma100'))
fig.add_trace(go.Scatter(x=data.Date, y=ma200,
                        mode='lines', name='Stock_Close with ma100'))


fig.update_layout(title='Stock Price Data with ma100 and ma200 2010-2021',
                 xaxis_title='Date', yaxis_title='Price',
                 xaxis_rangeslider_visible=True)
st.plotly_chart(fig)   


# Create a new dataframe with only the 'Close column'
data = data.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .75 ))

# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)


#load my model
model = load_model('keras_model.h8')


#Testing Part
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

st.subheader('Predicted Value vs Stock Close data Value')
st.write(valid)


st.subheader('Trained data vs closing data vs Predicted data')
fig2 = plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='upper left', fontsize='x-large')
st.pyplot(fig2)