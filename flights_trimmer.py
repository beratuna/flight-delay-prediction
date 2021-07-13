import pandas as pd
import numpy as np

def read_airports_data():
    return pd.read_csv("./dataset/flights/airports.csv")

# Reading the intersection of cities from txt file
def read_intersection():
    file = open('./dataset/intersection.txt', 'r')
    inter = file.read().split('\n')
    file.close()
    return inter

def read_flights_cities():
    file = open('./dataset/flights_cities.txt', 'r')
    cities = file.read().split('\n')
    file.close()
    return cities

def read_weather_cities():
    f = open('./dataset/weather_cities.txt', 'r')
    weather_cities = f.read().split('\n')
    f.close()
    return weather_cities

def get_flights_cities():
    airports_data = read_airports_data()
    flights_cities = airports_data["CITY"].tolist()
    flights_cities_set = set(flights_cities)
    # print(flights_cities_set)
    return flights_cities_set

# Finding the intersection of two datasets
def get_intersection():
    weather_cities = read_weather_cities()
    flights_cities = read_flights_cities()
    intersection = sorted(set(weather_cities).intersection(flights_cities))
    # print(intersection)
    return intersection

# Extracting the intersection of cities in between two datasets
def write_intersection():
    intersection = get_intersection()
    with open('./dataset/intersection.txt', 'w') as file:
        for item in intersection:
            if item == intersection[-1]:
                file.write("%s" % item)
            else:
                file.write("%s\n" % item)
    file.close()

def write_flights_cities():
    flights_cities = sorted(get_flights_cities())
    with open('./dataset/flights_cities.txt', 'w') as file:
        for item in flights_cities:
            if item == flights_cities[-1]:
                file.write("%s" % item)
            else:
                file.write("%s\n" % item)
    file.close()

def create_new_airports():
    intersection = read_intersection()
#    print(intersection)
    airports_data = read_airports_data()
    new_data = airports_data.drop(["AIRPORT", "STATE", "COUNTRY", "LONGITUDE", "LATITUDE"], axis=1)
    new_data = new_data[new_data.CITY.isin(intersection)]  # removing non-matching cities
    new_data.to_csv("./dataset/modified_flights/airports.csv", index=False, encoding="utf-8")

def get_iata():
    dict = {}
    df = pd.read_csv("./dataset/modified_flights/airports.csv")
    for index, row in df.iterrows():
        dict[row[0]] = row[1]
#        print(row[0] + " " + row[1])
    return dict
 
def polish_flights():
    list_fly = ['YEAR', 'DAY_OF_WEEK', 'AIRLINE',
                'FLIGHT_NUMBER', 'TAIL_NUMBER', 'DEPARTURE_TIME',
                'TAXI_OUT', 'WHEELS_OFF', 'SCHEDULED_TIME', 'ELAPSED_TIME',
                'AIR_TIME', 'DISTANCE', 'WHEELS_ON', 'TAXI_IN', 'ARRIVAL_TIME',
                'DIVERTED', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY',
                'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY']

    df = pd.read_csv("./dataset/flights/flights.csv", low_memory=False)
#    df.to_csv("./dataset/flights/flights.csv", index=True, encoding="utf-8")
#    df = pd.read_csv("./dataset/flights/flights.csv", low_memory=False)
#    
    new_data = df.drop(list_fly, axis=1)
    
    col = list(new_data.columns.values)
    new_data = new_data[new_data.CANCELLED == 0]  # removing non-matching cities
    new_data = new_data.drop(['CANCELLED'], axis = 1)
    
    iata_dic = get_iata()
    keys = iata_dic.keys()
    new_data = new_data[new_data.ORIGIN_AIRPORT.isin(keys)]  # removing non-matching cities
    new_data.ORIGIN_AIRPORT = new_data.ORIGIN_AIRPORT.map(lambda x: iata_dic[x])
    
    new_data = new_data[new_data.DESTINATION_AIRPORT.isin(keys)]  # removing non-matching cities
    new_data.DESTINATION_AIRPORT = new_data.DESTINATION_AIRPORT.map(lambda x: iata_dic[x])
    
#    delete 12-31
#    months = list(range(1, 12))
#    days = list(range(1, 31))
    months = [1,2,3,4,5,6,7,8,9,10,11]
    days =  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    new_data = new_data[(new_data.MONTH.isin(months)) | (new_data.DAY.isin(days))]

    def calc_time(t):
        time = int(t)
        min = time % 100
        if min <= 30:
            time = time - min
        else:
            time += 100 - min
        return time

    new_data.SCHEDULED_DEPARTURE = new_data.SCHEDULED_DEPARTURE.map(calc_time)
    new_data.SCHEDULED_ARRIVAL = new_data.SCHEDULED_ARRIVAL.map(calc_time)
    
    new_data.to_csv("./dataset/modified_flights/flights.csv", index=False, encoding="utf-8")
#    print(col)

def flight_time():
    df = pd.read_csv("./dataset/modified_flights/flights.csv", low_memory=False)
    def map_flight_dept(mon,day,time):
        time = time/100
        if(time == 24):
            mon, day = check_ends(mon, day)
            day += 1
            time = 0.0
        return str(mon) + "-" + str(day) + "|" + str(time)
        
    def check_ends(mon, day):
#        check for months
        if(mon == 1 or mon == 3 or mon == 5 or mon == 7 or mon == 8 or mon == 10 or mon == 12):
            if day == 31:
                day = 0
                mon += 1
        elif(mon == 4 or mon == 6 or mon == 9 or mon == 11):
            if day == 30:
                day = 0
                mon += 1
        elif(mon == 2):
            if day == 28:
                day = 0
                mon += 1                
                                              
        return mon, day


    def map_flight_arrv(mon,day,dep_time, arr_time):
        arr_time = arr_time/100
        dep_time = dep_time/100
        if(arr_time == 24):
            mon, day = check_ends(mon, day)
            arr_time = 0.0
            day += 1
        elif(arr_time < dep_time):
            mon, day = check_ends(mon, day)
            day += 1
        return str(mon) + "-" + str(day) + "|" + str(arr_time)
        
    df['SCHEDULED_ARRIVAL'] = df.apply(lambda x: map_flight_arrv(x['MONTH'], x['DAY'],
                                              x['SCHEDULED_DEPARTURE'], x['SCHEDULED_ARRIVAL']), axis=1)
    df['SCHEDULED_DEPARTURE'] = df.apply(lambda x: map_flight_dept(x['MONTH'], x['DAY'], x['SCHEDULED_DEPARTURE']), axis=1) 

    df = df.drop(['MONTH', 'DAY'], axis = 1)    
    
    df.to_csv("./dataset/modified_flights/flights.csv", index=False, encoding="utf-8")

# iata_dict = get_iata()
#write_flights_cities()
#write_intersection()
#create_new_dataset()
# polish_flights()
#get_iata()
#polish_flights()
#modify_time()
# flight_time()

#1508289#5805945,12,30,Phoenix,Seattle,2400,33.0,200


