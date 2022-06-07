#save_preprocessed_data should be run once to generate the preprocessed data and afterwards load_preprocessed_data can be run to get the data


import pandas as pd
import csv, sys
from matplotlib import pyplot as plt
import shapefile as shp
import numpy as np
from datetime import datetime as dt

def read_taxi_zones_shape_file(filename):
	shapefile = shp.Reader(filename)
	return shapefile



def plot_taxi_zones(taxi_zones_shape_data, center_points):
	plt.figure()
	for shape, center in zip(taxi_zones_shape_data.shapeRecords(), center_points):
		x = [i[0] for i in shape.shape.points[:]]
		y = [i[1] for i in shape.shape.points[:]]
		plt.plot(x, y)

		center_point = center[1]
		plt.plot(center_point[0],center_point[1],'*')

	plt.show()


def determine_centers_of_districts(taxi_zones_shape_data):
	#this determines an approximate center of the district, so it is not perfect
	# for rectangular/circular shapes it will be quite accurate, but for more irregular shapes
	# it could turn out to be inaccurate
	#centers = [(locationID, center_point),...,()]
	centers = []
	for shape in taxi_zones_shape_data.shapeRecords():
		x = [i[0] for i in shape.shape.points[:]]
		y = [i[1] for i in shape.shape.points[:]]

		x1 = min(x)
		x2 = max(x)

		y1 = min(y)
		y2 = max(y)

		center_point = (x1+(x2-x1)/2,y1+(y2-y1)/2)
		locationID = int(shape.record[1])

		centers.append((locationID, center_point))

	centers.sort()

	return centers

def preprocess_trip_data(trip_db, center_points, start_month_string, end_month_string):
	# process data from trip_db
	# get relevant data from preprocessed and split data
	# feature = time hour, day of the week, pointx, pointy, #ToDo: fill in

	proc_trip_data = []
	# sort based on pickup time
	trip_db = trip_db.sort_values('tpep_pickup_datetime')

	start_year = int(start_month_string[:4])
	start_month = int(start_month_string[5:])

	end_year = int(end_month_string[:4])
	end_month = int(end_month_string[5:])


	#ToDo: remove data points started before the start month or after the end month
	for datapoint in trip_db.iloc:
		timestamp = datapoint.tpep_pickup_datetime
		month = timestamp.month
		year = timestamp.year
		if end_year >= year >= start_year and end_month >= month >= start_month:
			hour_of_day = timestamp.hour
			day_of_week = timestamp.dayofweek

			pu_loc_id = datapoint.PULocationID
			try:
				pu_xy = center_points[pu_loc_id-1][1]
			except IndexError:
				#district not in center_points
				continue

			pu_x = pu_xy[0]
			pu_y = pu_xy[1]

			proc_data_point = np.array([hour_of_day, day_of_week, pu_x, pu_y])

			proc_trip_data.append(proc_data_point)


	proc_trip_data = np.array(proc_trip_data)

	return proc_trip_data




# def split_data_into_timeslots(timeslot_length, trip_db):
# 	#timeslot length in minutes
# 	#split based on pickup time
#
# 	pass



def save_preprocessed_data():
	filename = 'taxis.parquet'

	trip_db = pd.read_parquet(filename)

	taxi_zones_shape_data = read_taxi_zones_shape_file('taxi_zones_shape_files/taxi_zones.shp')

	center_points = determine_centers_of_districts(taxi_zones_shape_data)

	start_month = '2020-01'
	end_month = '2020-01'
	proc_trip_data = preprocess_trip_data(trip_db, center_points, start_month, end_month)
	np.save('processed_trip_data_1', proc_trip_data)

def load_preprocessed_data():
	return np.load('processed_trip_data_1.npy')



if __name__ == '__main__':
	# filename = 'yellow_tripdata_2022-01.parquet'
	# filename = 'yellow_tripdata_2020-01.parquet'
	# filename = 'fhv_tripdata_2022-01.parquet'

	#plotting stuff
	filename = 'taxis.parquet'
	trip_db = pd.read_parquet(filename)
	taxi_zones_shape_data = read_taxi_zones_shape_file('taxi_zones_shape_files/taxi_zones.shp')
	center_points = determine_centers_of_districts(taxi_zones_shape_data)
	# plot_taxi_zones(taxi_zones_shape_data, center_points)






	# timeslot_data = split_data_into_timeslots(timeslot_length, trip_db)


