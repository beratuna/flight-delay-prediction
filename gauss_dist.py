import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os


def gaus(feature):

    data = genfromtxt('./dataset/modified_weather/'+ feature +'.csv', delimiter=',')
    data = data.ravel()
    # clear nan values
    data = data[~np.isnan(data)]
    data = data.tolist()
    data = sorted(data)
    df = pd.DataFrame(data)
    a = df.describe()
    max_bound = (int(a.loc["max"]))
    upper_mid = (int(a.loc["75%"]))
    middle = (int(a.loc["50%"]))
    lower_mid = (int(a.loc["25%"]))
    min_bound = (int(a.loc["min"]))
    
    df = pd.read_csv('./dataset/modified_weather/' + feature +'.csv')

    # Change csv file inputs with labels, excluding datetime (first column)
    for i in range(1, len(df.columns)):
        city = df.columns[i]
#        print(city)
        boo1 = (df[city] >= min_bound) & (df[city] < lower_mid)
        boo2 = (df[city] >= lower_mid) & (df[city] < middle)
        boo3 = (df[city] >= middle) & (df[city] < upper_mid)
        boo4 = (df[city] >= upper_mid) & (df[city] <= max_bound)
        
        df.loc[boo1, [df.columns[i]]] = 1
        df.loc[boo2, [df.columns[i]]] = 2
        df.loc[boo3, [df.columns[i]]] = 3
        df.loc[boo4, [df.columns[i]]] = 4
        
    if not os.path.exists("./dataset/modified_weather/labelled_weather"):
        os.makedirs("./dataset/modified_weather/labelled_weather")
    df.to_csv('./dataset/modified_weather/labelled_weather/' + feature + '.csv', index=False, encoding="utf-8")

    return


def restore_desc():
    df = pd.read_csv("./dataset/modified_weather/weather_description.csv", delimiter=',')
    columns = df.columns[1:]
    
    s = set()
    for index, row in df.iterrows():
        for i in columns:
            s.add(row[i])
    dic = {}
    count = 1
    for i in s:
        dic[i] = count
        count += 1
    
    for col in columns:
        df[col] = df[col].map(dic)

    if not os.path.exists("./dataset/modified_weather/labelled_weather"):
        os.makedirs("./dataset/modified_weather/labelled_weather")
    df.to_csv('./dataset/modified_weather/labelled_weather/weather_description.csv', index=False, encoding="utf-8")


def generate_labelled():
    gaus("humidity")
    gaus("pressure")
    gaus("temperature")
    gaus("wind_speed")
    restore_desc()
