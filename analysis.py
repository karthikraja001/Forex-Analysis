from report import generateReport
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime
import argparse
import charts

def fetchLongData(currency):
    return yf.Ticker(currency)

def fetch10YData(currency):
    return yf.download(currency, start="2010-01-01")

def regressionModel(df):
    df.reset_index(inplace = True)
    df.ffill(inplace = True)
    regDf = df.copy()
    regDf = regDf[['Adj Close']]
    predict_days = 7
    regDf['Prediction'] = regDf[['Adj Close']].shift(-predict_days)

    x = np.array(regDf.drop(['Prediction'], 1)) #Takes only Adj Close
    x = x[:-predict_days]
    y = np.array(regDf['Prediction']) # takes Prediction
    y = y[:-predict_days]

    X_mean = np.nanmean(x) #Actual Price
    Y_mean = np.nanmean(y) #Shifted Price
    num = 0
    den = 0
    for i in range(len(x)):
        num += (x[i] - X_mean)*(y[i] - Y_mean)
        den += (x[i] - X_mean)**2 # Square Value
    m = num / den
    c = Y_mean - m*X_mean
    print("M = "+str(m))
    print("C = "+str(c))
    datelist, xPred = findForecast(regDf,predict_days,m,c)
    return datelist,xPred

def findForecast(regDf,predict_days,m,c):
    x_forecast = np.array(regDf.drop(['Prediction'],1))[-predict_days:]

    xPred = []
    for abc in x_forecast:
        v = float(abc*m+c)
        xPred.append(v)
    
    se = 0
    for i in range(predict_days):
        se += (xPred[i] - x_forecast[i]) ** 2

    mse = se / predict_days
    rmse = np.sqrt(mse)
    print("RMSE = "+str(rmse))

    datelist = pd.date_range(datetime.today(), periods=100).tolist()
    return datelist, xPred

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=FutureWarning)
    my_parser = argparse.ArgumentParser(allow_abbrev=False)
    my_parser.add_argument('--currency', action='store', type=str, required=True)
    my_parser.add_argument('--output', action='store', type=str, required=True)
    args = my_parser.parse_args()
    currency = args.currency
    outPath = args.output
    inr = fetchLongData(currency)
    df = fetch10YData(currency)
    dfInfo = inr.info
    datelist, xPred = regressionModel(df)
    fig, newFig = charts.smaChart(df)
    theFig, yesFig = charts.yesterday()
    weeklyFig = charts.weeklyChart(dfInfo)
    monthlyFig =  charts.monthlyChart(dfInfo)
    generateReport(dfInfo, weeklyFig, monthlyFig, newFig, df, datelist, xPred, fig, yesFig, theFig, outPath)
    print("Report Generated Successfully")