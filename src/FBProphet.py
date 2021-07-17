import numpy as np
import pandas as pd
import plotly.graph_objects as go
import fbprophet as prophet
from PreProcessing import get_dates


def fbprophet(df, dates):
    df = df.reset_index()
    final_df = df[['Date', 'Close']]
    final_df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
    model = prophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode="additive")
    model.fit(final_df)
    close_prices = pd.DataFrame(data=dates, columns=['ds'])
    forecast = model.predict(close_prices)
    data = {'Date': dates, 'Close': forecast['yhat']}
    df_temp = pd.DataFrame(data=data)
    final_df.rename(columns={'ds': 'Date', 'y': 'Close'}, inplace=True)
    result_df = pd.concat([final_df, df_temp])
    result_df = result_df.reset_index(drop=True)
    return result_df


def fbprophet_accuracy(df):
    df = df.reset_index()
    final_df = df[['Date', 'Close']]
    final_df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
    train = final_df[:len(final_df) - 100]
    valid = final_df[len(df) - 100:]
    model = prophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode="additive")
    model.fit(train)
    close_prices = model.make_future_dataframe(periods=100)
    forecast = model.predict(close_prices)
    forecast_valid = forecast['yhat'][len(final_df) - 100:]
    rms = np.sqrt(np.mean(np.power((np.array(valid['y']) - np.array(forecast_valid)), 2)))
    return rms


def fbprophet_graph(df):
    final_df = fbprophet(df, get_dates(df))

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
            'text': "FBProphet Prediciton",
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

    return fig
