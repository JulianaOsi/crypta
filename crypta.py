import requests
import csv
import numpy
from datetime import datetime

import pandas as pd

from_symbol = 'BTC'
to_symbol = 'USD'
exchange = 'Bitstamp'
datetime_interval = 'day'

def get_filename(from_symbol, to_symbol, exchange, datetime_interval, download_date):
    return '%s_%s_%s_%s_%s.csv' % (from_symbol, to_symbol, exchange, datetime_interval, download_date)


def download_data(from_symbol, to_symbol, exchange, datetime_interval):
    supported_intervals = {'minute', 'hour', 'day'}
    assert datetime_interval in supported_intervals,\
        'datetime_interval should be one of %s' % supported_intervals

    print('Downloading %s trading data for %s %s from %s' %
          (datetime_interval, from_symbol, to_symbol, exchange))
    base_url = 'https://min-api.cryptocompare.com/data/histo'
    url = '%s%s' % (base_url, datetime_interval)

    params = {'fsym': from_symbol, 'tsym': to_symbol,
              'limit': 2000, 'aggregate': 1,
              'e': exchange}
    request = requests.get(url, params=params)
    data = request.json()
    return data


def convert_to_dataframe(data):
    df = pd.io.json.json_normalize(data, ['Data'])
    df['datetime'] = pd.to_datetime(df.time, unit='s')
    df = df[['datetime', 'low', 'high', 'open',
             'close', 'volumefrom', 'volumeto']]
    return df


def filter_empty_datapoints(df):
    indices = df[df.sum(axis=1) == 0].index
    print('Filtering %d empty datapoints' % indices.shape[0])
    df = df.drop(indices)
    return df


data = download_data(from_symbol, to_symbol, exchange, datetime_interval)
df = convert_to_dataframe(data)
df = filter_empty_datapoints(df)

current_datetime = datetime.now().date().isoformat()
filename = get_filename(from_symbol, to_symbol, exchange, datetime_interval, current_datetime)
print('Saving data to %s' % filename)
df.to_csv(filename, index=False, sep=';')

def read_dataset(filename):
    print('Reading data from %s' % filename)
    df = pd.read_csv(filename, sep=';')
    df.datetime = pd.to_datetime(df.datetime) # change type from object to datetime
    df = df.set_index('datetime')
    df = df.sort_index() # sort by datetime
    print(df.shape)
    return df

df = read_dataset(filename)

from stockstats import StockDataFrame
df = StockDataFrame.retype(df)
df['macd'] = df.get('macd') # calculate MACD
df['rsi_14'] = df.get('rsi_14')
df['wr_14'] = df.get('wr_14')
df['kdjd3'] = df.get('kdjd_3')
df['kdjk14'] = df.get('kdjk_14')

#########################################

df['close_15_sma'] = df.get('close_15_sma')

df.to_csv("indicators.csv", index=True, sep=';')


def read_column(filename, header):
    column = pd.read_csv(filename, sep=';')
    return column[header]

def sma_trend(column, index):
    counter_rise = 0
    counter_fall = 0

    for i in range(4):
        if column[index] < column[index + 1]:
            counter_fall = 0
            counter_rise += 1
        elif column[index] > column[index + 1]:
            counter_rise = 0
            counter_fall += 1
        elif column[index] == column[index + 1]:
            counter_rise += 1
            counter_fall += 1
        index += 1
    if counter_rise == 4 and counter_fall != 4:
        return 1
    elif counter_fall == 4 and counter_rise != 4:
        return -1
    else:
        return 0

def close_trend(column_close, column_sma, index):
    close_leads = 0
    close_lags = 0

    for i in range(5):
        if column_close[index] < column_sma[index]:
            close_leads = 0
            close_lags += 1
        elif column_close[index] > column_sma[index]:
            close_lags = 0
            close_leads += 1
        elif column_close[index] == column_sma[index]:
            close_lags += 1
            close_leads += 1
        index += 1
    if close_leads == 5 and close_lags != 5:
        return 1
    elif close_lags == 5 and close_leads != 5:
        return -1
    else:
        return 0

def get_trends(closing_price, close_15_sma):
    result = []
    for index in range(len(closing_price)):
        if index == 0:
            index = 4
        if close_trend(closing_price, close_15_sma, index - 4) == 1 and sma_trend(close_15_sma, index - 4) == 1:
            result.append('up')
        elif close_trend(closing_price, close_15_sma, index - 4) == -1 and sma_trend(close_15_sma, index - 4) == -1:
            result.append('down')
        else:
            result.append('no')
    return result

def get_trading_signals(closing_price, trends):
    data = normalize_data(closing_price)
    last_trend = ''
    for i in range(len(trends)):
        data[i] *= 0.5
        if trends[i] == 'up':
            last_trend = 'up'
            data[i] += 0.5
        elif trends[i] == 'down':
            last_trend = 'down'
        elif i > 0 and trends[i] == 'no' and last_trend == 'up':
            data[i] += 0.5
    return data

def normalize_data(data):
    result = []
    for i in range(len(data)):
        cp = []
        for a in range(3):
            if i + a < len(data):
                cp.append(data[i + a])
        if max(cp) - min(cp) == 0.0:
            result.append(0.0)
            continue
        value = (data[i] - min(cp))/(max(cp) - min(cp))
        if numpy.isnan(value):
            result.append(0.0)
        else:
            result.append(value)
    return result

def write_data_to_csv(filename, columns_names, data, rows_amount):
    with open(filename, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns_names, delimiter=";", lineterminator='\n')
        writer.writeheader()
        for i in range(rows_amount):
            d = {}
            for index in range(len(columns_names)):
                pair = {columns_names[index]: data[index][i]}
                d.update(pair)

            writer.writerow(d)

import datetime


open_price = read_column('indicators.csv', 'open').get_values()
close = read_column('indicators.csv', 'close').get_values()
close_15_sma = read_column('indicators.csv', 'close_15_sma').get_values()
trends = get_trends(close, close_15_sma)
trading_signals = get_trading_signals(close, trends)

ma_15 = normalize_data(close_15_sma)
macd = normalize_data(read_column('indicators.csv', 'macd').get_values())
rsi_14 = normalize_data(read_column('indicators.csv', 'rsi_14').get_values())
wr_14 = normalize_data(read_column('indicators.csv', 'wr_14').get_values())
d_3 = normalize_data(read_column('indicators.csv', 'kdjd3').get_values())
k_14 = normalize_data(read_column('indicators.csv', 'kdjk14').get_values())

numbers_of_rows = numpy.arange(1, len(close)+1)

data = [numbers_of_rows,
        close,
        ma_15,
        macd,
        rsi_14,
        wr_14,
        d_3,
        k_14,
        trends,
        trading_signals]

columns = ["Time series", "Closing price", "MA-15", "MACD-26", "RSI-14", "WR-14", "D-3", "K-14", "Trend", "Trading signals"]

write_data_to_csv("data.csv", columns, data, len(numbers_of_rows))

#--------ML-------------
from numpy import array

x = [ma_15, macd, rsi_14, wr_14, d_3, k_14]  # Features
X=array(x)
X=X.transpose()
y = trading_signals  # Target variable

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

start_date = 31

X_train = array([ma_15[0:-start_date], macd[0:-start_date], rsi_14[0:-start_date], wr_14[0:-start_date], d_3[0:-start_date], k_14[0:-start_date]]).transpose()
y_train = y[0:-start_date]
X_test = array([ma_15[-start_date:], macd[-start_date:], rsi_14[-start_date:], wr_14[-start_date:], d_3[-start_date:], k_14[-start_date:]]).transpose()

#print(X_train)
#print(y_train)
#print(X_test)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, min_samples_split=2)

regressor.fit(X_train, y_train)  # fit the model with data
y_pred = regressor.predict(X_test)

new_y = [y_pred]

#отчет о точности
from sklearn.metrics import mean_squared_error
#print((mean_squared_error(y_test, y_pred)))

def get_predicted_trends(OTr, meanTr):
    result = []
    for i in range(len(OTr)):
        if OTr[i] > meanTr:
            result.append('Up')
        else:
            result.append('Down')
    return result

def get_trading_decision(trends):
    result = []
    result.append('Hold')
    for i in range(len(trends) - 1):
        currTrend = trends[i]
        if currTrend == trends[i + 1]:
            result.append('Hold')
        elif currTrend != trends[i + 1] and trends[i + 1] == 'Up':
            result.append('Buy')
        elif currTrend != trends[i + 1] and trends[i + 1] == 'Down':
            result.append('Sell')
    return result

tr_mean = numpy.mean(trading_signals[-start_date:])
print(tr_mean)
predictedTrends = get_predicted_trends(y_pred, tr_mean)
tradingDecisions = get_trading_decision(predictedTrends)
rows = numpy.arange(1, len(y_pred)+1)
base = datetime.date.today() - datetime.timedelta(days=start_date)
date_list = [base + datetime.timedelta(days=x) for x in range(0, start_date)]

columns = ["Time series", "Date", "OTr", "Trend", "Trading decision"]
d = [rows, date_list, y_pred, predictedTrends, tradingDecisions]

write_data_to_csv("OTri.csv", columns, d, len(y_pred))

start_balance_USD = 1500
curr_balance_USD = 1500
curr_balance_BTC = 0
BTC_to_USD_rate = open_price[-start_date:]

print("BTC/USD", datetime.date.today(), ': ', BTC_to_USD_rate[start_date - 1])

def get_profit_for_komozzkii(trading_decisions, start_balance, curr_balance_usd, curr_balance_btc, btc_to_usd_rate):
    for i in range(len(trading_decisions)):
        if trading_decisions[i] == 'Buy':
            curr_balance_btc += curr_balance_usd / btc_to_usd_rate[i]
            curr_balance_usd = 0
        if trading_decisions[i] == 'Sell':
            curr_balance_usd += curr_balance_btc * btc_to_usd_rate[i]
            curr_balance_btc = 0

    if curr_balance_btc == 0:
        print("USD profit", curr_balance_usd - start_balance)
    else:
        print("BTC profit", curr_balance_btc - start_balance / btc_to_usd_rate[0])

get_profit_for_komozzkii(tradingDecisions, start_balance_USD, curr_balance_USD, curr_balance_BTC, BTC_to_USD_rate)

