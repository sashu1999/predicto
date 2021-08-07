import pandas as pd
import plotly.graph_objects as go
from PreProcessing import get_dates, expand_test_dataset, expand_train_dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import math

pd.options.mode.chained_assignment = None  # default='warn'


def linear_regression(df, dates):
    df = df.reset_index()
    final_df = df[['Date', 'Close']]
    new_data = expand_train_dataset(df)
    train_y = new_data['Close']
    new_data.drop('Close', axis=1, inplace=True)
    train_x = new_data
    pipeline = Pipeline(steps=[('normalize', StandardScaler()), ('model', LinearRegression())])
    pipeline.fit(train_x, train_y)
    df = expand_test_dataset(dates)
    x = pipeline.predict(df)
    data = {'Date': dates, 'Close': x}
    df_temp = pd.DataFrame(data=data)
    result_df = pd.concat([final_df, df_temp])
    result_df = result_df.reset_index()
    return result_df


def linear_regression_graph(df):
    final_df = linear_regression(df, get_dates(df))

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
            'text': "Linear Regression Prediciton",
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


def linear_regression_accuracy(df):
    df = df.reset_index()
    new_data = expand_train_dataset(df)

    y = new_data['Close']
    new_data.drop('Close', axis=1, inplace=True)
    X = new_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=False)
    pipeline = Pipeline(steps=[('normalize', StandardScaler()), ('model', LinearRegression())])
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)

    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
    return error_lr
