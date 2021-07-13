import pandas as pd

city_attr_data = pd.read_csv("./dataset/weather/city_attributes.csv")
humid_data = pd.read_csv("./dataset/weather/humidity.csv")
press_data = pd.read_csv("./dataset/weather/pressure.csv")
temp_data = pd.read_csv("./dataset/weather/temperature.csv")
desc_data = pd.read_csv("./dataset/weather/weather_description.csv")
windSpeed_data = pd.read_csv("./dataset/weather/wind_speed.csv")

# List of all datas related to weather
data = [humid_data, press_data, temp_data, desc_data, windSpeed_data]
features = ["humidity", "pressure", "temperature", "weather_description", "wind_speed"]

# Extracting US cities
def get_US_cities():
    US_cities_data = city_attr_data.loc[(city_attr_data["Country"] == "United States")]
    US_cities = US_cities_data.City.tolist()
    # print(US_cities)
    return US_cities

# Extracting non-US cities and converting to a list
def get_nonUS_cities():
    nonUS_cities_data = city_attr_data.loc[(city_attr_data["Country"] != "United States")]
    nonUS_cities = nonUS_cities_data.City.tolist()
    # Appending the only non-matching city with flights dataset
    nonUS_cities.append("Saint Louis")
    # print(nonUS_cities)
    return nonUS_cities

def find_non2015():
    non_2015_data = humid_data[humid_data.datetime.str.find("2015") != -1]
    print(non_2015_data.datetime.tolist())

# Writing the US cities into a txt file
def write_weather_cities():
    cities = sorted(get_US_cities())
    with open('./dataset/weather_cities.txt', 'w') as file:
        for item in cities:
            if item == cities[-1]:
                file.write("%s" % item)
            else:
                file.write("%s\n" % item)
    file.close()

# Reading the intersection of cities from txt file
def read_intersection():
    file = open('./dataset/intersection.txt', 'r')
    inter = file.read().split('\n')
    file.close()
    return inter

# Creating modified datasets in a new folder
def create_new_dataset():
    for i in range(0, len(data)):
        new_data = data[i].drop(get_nonUS_cities(), axis=1)  # removing nonUS cities from all
        new_data = new_data[new_data.datetime.str.find("2015") != -1]  # removing non2015 data
        new_data.to_csv("./dataset/modified_weather/" + features[i] + ".csv", index=False, encoding="utf-8")
