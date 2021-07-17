import numpy as np
import pandas as pd
from pmdarima import auto_arima
import plotly.graph_objects as go
from PreProcessing import get_dates


def arima_accuracy(df):
    df = df.reset_index()
    df = df[['Date', 'Close']]

    train = df[:len(df) - 100]
    valid = df[len(df) - 100:]

    training = train['Close']
    model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                       error_action='ignore', suppress_warnings=True)
    model.fit(training)

    forecast = model.predict(n_periods=100)
    forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

    rms = np.sqrt(np.mean(np.power((np.array(valid['Close']) - np.array(forecast['Prediction'])), 2)))
    return rms


def arima(df, dates):
    df = df.reset_index()
    final_df = df[['Date', 'Close']]
    training = final_df['Close']
    model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                       error_action='ignore', suppress_warnings=True)
    model.fit(training)

    forecast = model.predict(n_periods=100)
    data = {'Date': dates, 'Close': forecast}
    df_temp = pd.DataFrame(data=data)
    result_df = pd.concat([final_df, df_temp])
    result_df = result_df.reset_index()
    return result_df


def arima_graph(df):
    final_df = arima(df, get_dates(df))

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
            'text': "Arima Prediciton",
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