import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from read_data import read_taxi_zones_shape_file, determine_centers_of_districts


verbose = False
plotting = True
clusters = 4

week_days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}


dataset = np.load('processed_trip_data_2.npy')
data = pd.DataFrame(dataset, columns=['hour', 'day', 'x_loc', 'y_loc', 'passengers', 'duration'])

taxi_zones_shape_data = read_taxi_zones_shape_file('taxi_zones_shape_files/taxi_zones.shp')
center_points = determine_centers_of_districts(taxi_zones_shape_data)


# Determine how many hours a time slot consists of
slot_size = 3  # hours
slots_per_day = 24 // slot_size
slots_per_week = slots_per_day * 7

# Check validity of the time slot size
assert type(slot_size) == int
assert int(slots_per_day * slot_size) == 24


# Time it
start = time.time()
fig, axes = None, None

# Iterate through each time slot
for i in range(slots_per_week):
    # Get the data corresponding to this time slot
    day = i // slots_per_day
    start_hour = (i % slots_per_day) * slot_size
    end_hour = start_hour + slot_size - 1

    data_slot = data[np.logical_and(data['day'] == day,
                                    np.logical_and(start_hour <= data['hour'], data['hour'] <= end_hour))]
    data_slot_loc = data_slot[['x_loc', 'y_loc']]

    # Plot
    if start_hour == 0:  # start_hour == 0:
        fig, axes = plt.subplots(2, 12//slot_size, figsize=(5 * 12//slot_size, 10))
        axes = np.array(axes).flatten()

    if plotting:
        for shape in taxi_zones_shape_data.shapeRecords():
            x_shape = [i[0] for i in shape.shape.points[:]]
            y_shape = [i[1] for i in shape.shape.points[:]]
            axes[start_hour//slot_size].scatter(x_shape, y_shape, s=.1, c='lightgrey')

        # plot LaGuardia and JFK Airport
        axes[start_hour//slot_size].scatter(-73.876, 40.774, s=120, marker='+', c='green')  # LaGuardia
        axes[start_hour//slot_size].scatter(-73.790, 40.645, s=120, marker='+', c='blue')   # JFK

        if verbose:
            for _, center in center_points:
                d = data_slot[np.logical_and(center[0] == data_slot['x_loc'], center[1] == data_slot['y_loc'])]
                dx, dy = np.array(d['x_loc']), np.array(d['y_loc'])
                len(dx) != 0 and axes[start_hour//slot_size].scatter(dx[0], dy[0], s=len(d['x_loc']) / 50, c='black')

    # Do K-Means and plot the centers
    k_means = KMeans(n_clusters=clusters).fit(data_slot_loc)
    centroids = k_means.cluster_centers_
    [axes[start_hour//slot_size].scatter(x_c, y_c, c='red') for [x_c, y_c] in centroids]
    axes[start_hour//slot_size].set_title(f"Hour {start_hour} - {end_hour}")
    fig.suptitle(f"{week_days[day]}")
    print(f'Done: {week_days[day]}, Hour {start_hour} - {end_hour}')

    if end_hour == 23:
        fig.savefig(f"{day + 1}_{week_days[day]}")
        fig.show()
        print()
        # input('press enter')

print(f'Time elapsed: {round(time.time() - start, 3)} seconds.')
