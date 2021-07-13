import weather_trimmer as wt
import flights_trimmer as ft
import merger as m
import gauss_dist as g

wt.create_new_dataset()
wt.write_weather_cities()

ft.write_flights_cities()
ft.write_intersection()
ft.create_new_airports()
ft.polish_flights()
ft.flight_time()

g.generate_labelled()

m.merge_weather()
m.merge_flight_weather()
