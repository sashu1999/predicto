import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import plotly.graph_objects as go
import numpy as np
from PreProcessing import get_dates


def lstm_accuracy(df):
    df = df.reset_index()
    df = df[['Date', 'Close']]
    training_set = df.iloc[:len(df) - 100, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    y_train = []
    for i in range(60, len(df) - 100):
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=100, batch_size=32)

    df = df.reset_index()
    dataset_total = df[['Close']]

    inputs = dataset_total[len(dataset_total) - 160:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 160):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    targets = df[len(df) - 100:]['Close'].values
    predictions = predicted_stock_price

    rmse = np.sqrt(((predictions - targets) ** 2).mean())

    return rmse


def lstm(df, dates):
    df = df.reset_index()
    df = df[['Date', 'Close']]
    final_df = df[['Date', 'Close']]
    training_set = df.iloc[:, 1:2].values
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    y_train = []
    for i in range(60, len(df)):
        X_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=100, batch_size=32)

    df = df.reset_index()
    dataset_total = df[['Close']]

    inputs = dataset_total[len(dataset_total) - 160:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 160):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predict = []
    for i in range(len(predicted_stock_price)):
        predict.append(predicted_stock_price[i][0])
    data = {'Date': dates, 'Close': predict}
    df_temp = pd.DataFrame(data=data)
    result_df = pd.concat([final_df, df_temp])

    return result_df


def lstm_graph(df):
    final_df = lstm(df, get_dates(df))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=final_df.loc[len(final_df)-198:len(final_df)-99]['Date'], y=final_df.loc[len(final_df)-198:len(final_df)-99]['Close'],
                   mode='lines',
                   name='Current'))
    fig.add_trace(go.Scatter(x=final_df.loc[len(final_df)-100:]['Date'], y=final_df.loc[len(final_df)-100:]['Close'],
                             mode='lines',
                             name='Predicted'))

    fig.update_layout(
        title={
            'text': "LSTM Prediciton",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        )
    )

    return (fig)