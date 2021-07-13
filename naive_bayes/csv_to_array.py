from numpy import genfromtxt



my_data = genfromtxt('dataset/modified_weather/temperature.csv', delimiter=',')

print(my_data)
