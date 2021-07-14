import time
import requests
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def get_stock_data(symbol):
    symbol = symbol.upper()
    tday = int(time.time())
    period1 = str(tday - 63072000)
    period2 = str(tday)
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5E" + symbol + "?formatted=true&crumb=RvGu5ZmMrvW&lang=en-IN&region=IN&includeAdjustedClose=true&interval=1d&period1={}&period2={}&events=div%7Csplit&useYfid=true&corsDomain=in.finance.yahoo.com".format(
        period1, period2)
    payload = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=payload)
    if r.status_code != 200:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + "?formatted=true&crumb=RvGu5ZmMrvW&lang=en-IN&region=IN&includeAdjustedClose=true&interval=1d&period1={}&period2={}&events=div%7Csplit&useYfid=true&corsDomain=in.finance.yahoo.com".format(
            period1, period2)
        r = requests.get(url, headers=payload)
    timestamp = r.json()['chart']['result'][0]['timestamp']
    quote = r.json()['chart']['result'][0]['indicators']
    new_dict = dict()
    new_dict['Date'] = timestamp
    new_dict.update(quote['quote'][0])
    df = pd.DataFrame(new_dict)
    df['Date'] = pd.to_datetime(df['Date'], utc=False, unit='s')
    df['Date'] = df['Date'].dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[['Date', 'open', 'high', 'low', 'close', 'volume']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df.dropna(inplace=True)

    r.close()
    return df
