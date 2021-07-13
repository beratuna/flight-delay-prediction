# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 02:26:36 2018
@author: kbera
"""
import pandas as pd


# Reading the intersection of cities from txt file
def read_intersection():
    file = open('./dataset/intersection.txt', 'r')
    inter = file.read().split('\n')
    file.close()
    return inter

def merge_weather():
    # humid_data = pd.read_csv("./dataset/modified_weather/labelled_weather/humidity.csv")
    # press_data = pd.read_csv("./dataset/modified_weather/labelled_weather/pressure.csv")
    # temp_data = pd.read_csv("./dataset/modified_weather/labelled_weather/temperature.csv")
    # desc_data = pd.read_csv("./dataset/modified_weather/labelled_weather/weather_description.csv")
    # windSpeed_data = pd.read_csv("./dataset/modified_weather/labelled_weather/wind_speed.csv")

    humid_data = pd.read_csv("./dataset/modified_weather/labelled_weather/humidity.csv")
    press_data = pd.read_csv("./dataset/modified_weather/labelled_weather/pressure.csv")
    temp_data = pd.read_csv("./dataset/modified_weather/labelled_weather/temperature.csv")
    desc_data = pd.read_csv("./dataset/modified_weather/labelled_weather/weather_description.csv")
    windSpeed_data = pd.read_csv("./dataset/modified_weather/labelled_weather/wind_speed.csv")


    new_data = pd.DataFrame([], columns = ["datetime", "humid", "press", "temp", "desc", "wind", "city"])
    inter = read_intersection()
    
    for city in inter:
        
        hum = humid_data[['datetime', city]]
        press = press_data[['datetime', city]]
        temp = temp_data[['datetime', city]]
        desc = desc_data[['datetime', city]]
        wind = windSpeed_data[['datetime', city]]
        temp1 = pd.merge(hum, press, on ="datetime")
        temp1 = pd.merge(temp1, temp, on ="datetime")
        temp1 = pd.merge(temp1, desc, on ="datetime")
        temp1 = pd.merge(temp1, wind, on ="datetime")
        
        empty_city = []
        for i in range(len(hum)):
            empty_city.append(city)
            
        temp1['city'] = empty_city
        temp1.columns = ["datetime", "humid", "press", "temp", "desc", "wind", "city"]
        
        for index, row in temp1.iterrows():
            row[6] = city    
        
        new_data = new_data.append(temp1)
#        print(new_data)

    def restore_datetime(d):
        date = d.split(" ")
        time = date[1]
        date = date[0]

        t = time.split(':')
        hour = t[0]

        y = date.split('-')
        mon = y[1]
        day = y[2]

        res_col = str(int(mon)) + '-' + str(int(day)) + '|' + str(float(int(hour)))
        return res_col

    #Change datetime structure to "Mon-Day|Hour"
    new_data.datetime = new_data.datetime.map(restore_datetime)
    new_data = new_data.dropna()
    new_data.to_csv("./dataset/modified_weather/merged.csv", index=False, encoding="utf-8")


def merge_flight_weather():
    flights = pd.read_csv("./dataset/modified_flights/flights.csv")
    weather = pd.read_csv("./dataset/modified_weather/merged.csv")

    weather.columns = ["SCHEDULED_DEPARTURE","dep_humid", "dep_press", 
                           "dep_temp", "dep_desc", "dep_wind", "ORIGIN_AIRPORT"]

    mer_dep = pd.merge(flights, weather, on=["SCHEDULED_DEPARTURE", "ORIGIN_AIRPORT"])

    weather.columns = ["SCHEDULED_ARRIVAL","arr_humid", "arr_press", 
                       "arr_temp", "arr_desc", "arr_wind", "DESTINATION_AIRPORT"]
    
    mer_final = pd.merge(mer_dep, weather, on=["SCHEDULED_ARRIVAL", "DESTINATION_AIRPORT"])
#    mer_final = mer_final.dropna()
    mer_final = mer_final.drop(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL'], axis=1)
    mer_final.to_csv("./dataset/final.csv", index=False, encoding="utf-8")
#    print(mer_final.head(10))
