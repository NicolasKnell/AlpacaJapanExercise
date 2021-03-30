import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as r


# Creation of a pandas DataFrame from the initial list of tuples
def create_df(points):
    return pd.DataFrame(points, columns=['x', 'y'])


# Random generation of initial centroids. k is the number of clusters (or centroids) wanted.
def initialization(k, df_points):
    # Calculation of min and max to chose centroids not too far from
    minx = min(df_points['x'])
    miny = min(df_points['y'])
    maxx = max(df_points['x'])
    maxy = max(df_points['y'])
    initial_centroids = [[r.random()*(maxx-minx)+minx, r.random()*(maxy-miny)+miny] for _ in range(k)]
    return initial_centroids


# assignment of the points to the closest centroid. This function calculates the distance between every point and every
# centroid in order to find the closest.
def points_assignment(df_points, centroids):
    k = len(centroids)
    for i in range(k):
        # df_points[i] is the column corresponding to the distances between each point and the centroid "i"
        df_points[i] = np.sqrt((df_points['x'] - centroids[i][0]) ** 2 + (df_points['y'] - centroids[i][1]) ** 2)
    df_nearest = df_points[[i for i in range(k)]]  # Creating a DataFrame excluding the unwanted columns
    df_points['nearest_centroid'] = df_nearest.idxmin(axis=1)


# Once all the points are assigned to a centroid, we need to update the position of the centroids to the barycenter of
# the points belonging to it.
def centroids_update(df_points, centroids):
    for i in range(len(centroids)):
        centroids[i][0] = np.mean(df_points.loc[df_points['nearest_centroid'] == i, ['x']])[0]  # mean for x axis
        centroids[i][1] = np.mean(df_points.loc[df_points['nearest_centroid'] == i, ['y']])[0]  # means for y axis
    return centroids


# Function combining all the steps of the process and iterating until the points stop switching to another cluster.
def k_means(k, points):
    df_points = create_df(points)
    centroids = initialization(k, df_points)
    points_assignment(df_points, centroids)
    # loop will continue until there are no more differences between 2 steps
    while True:
        nearest_centroid = df_points['nearest_centroid'].copy(deep=True)
        centroids = centroids_update(df_points, centroids)
        points_assignment(df_points, centroids)
        if sum(df_points['nearest_centroid'] != nearest_centroid) == 0:
            break
    df_clusters = pd.DataFrame(centroids, columns=['x_axis', 'y_axis'])
    df_clusters['Cluster_id'] = [i for i in range(k)]
    return df_clusters, df_points[['x', 'y', 'nearest_centroid']]  # In addition to the set of clusters with centroids,
    # I chose to return the points with their nearest centroid in order to vizualize

