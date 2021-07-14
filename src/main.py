from flask import Flask
from flask import render_template, request
from LinearRegression import linear_regression_accuracy, linear_regression_graph
from FBProphet import fbprophet_accuracy, fbprophet_graph
from LSTM import lstm_accuracy, lstm_graph
from Arima import arima_graph, arima_accuracy
from StockData import get_stock_data
import plotly
import json
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__, static_url_path='/static')


@app.route("/")
@app.route('/index')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST", "GET"])
def predict():
    ticker = request.form["ticker"]
    algorithm = request.form['algorithm']
    alg = ['Linear Regression', 'FB Prophet', 'LSTM', 'Arima']
    if int(algorithm) == 1:
        df = get_stock_data(ticker)
        graph = linear_regression_graph(df)
        rmse = linear_regression_accuracy(df)
        graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template(
            "output.html",
            ticker=ticker.upper(),
            plot=graphJSON,
            error=rmse,
            alg=alg[int(algorithm) - 1]
        )
    elif int(algorithm) == 2:
        df = get_stock_data(ticker)
        graph = fbprophet_graph(df)
        rmse = fbprophet_accuracy(df)
        graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template(
            "output.html",
            ticker=ticker.upper(),
            plot=graphJSON,
            error=rmse,
            alg=alg[int(algorithm) - 1]
        )
    elif int(algorithm) == 3:
        df = get_stock_data(ticker)
        graph = lstm_graph(df)
        rmse = lstm_accuracy(df)
        graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template(
            "output.html",
            ticker=ticker.upper(),
            plot=graphJSON,
            error=rmse,
            alg=alg[int(algorithm) - 1]
        )
    elif int(algorithm) == 4:
        df = get_stock_data(ticker)
        graph = arima_graph(df)
        rmse = arima_accuracy(df)
        graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template(
            "output.html",
            ticker=ticker.upper(),
            plot=graphJSON,
            error=rmse,
            alg=alg[int(algorithm) - 1]
        )


app.run('127.0.0.1', port=5000)
