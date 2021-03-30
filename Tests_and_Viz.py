import k_Means_code as kmc
import matplotlib.pyplot as plt
import seaborn as sns

# Set of 2D points
points = [(1.1, 2.5), (3.4, 1.9), (4.9, 2.9), (0.1, 4.0), (1.9, 1.1), (3.2, 4.4), (2.2, 2.9), (1.9, 0.4), (4.4, 4.1),
          (4, 3.3), (2.9, 0.7), (1, 4), (4, 4), (0.5, 1)]

# Use of k_means function on these points with 3 clusters
df_cluster, df_points = kmc.k_means(3, points)

print(df_cluster) # set of all clusters with the position of centroids

# Commands to plot the results
fig, ax = plt.subplots()
scatter1 = ax.scatter(df_points['x'], df_points['y'], c=df_points['nearest_centroid'])
scatter2 = ax.scatter(df_cluster['x_axis'], df_cluster['y_axis'], c = df_cluster['Cluster_id'], marker = 'x')

legend1 = ax.legend(*scatter1.legend_elements(), loc="lower left", title="Clusters")
ax.add_artist(legend1)

legend2 = ax.legend(*scatter2.legend_elements(), loc="lower right", title="Centroids")

ax.set_xlim(0,5)
ax.set_ylim(0,5)
ax.set_xlabel('x_axis')
ax.set_ylabel('y_axis')
ax.set_title('Points and Clusters')

plt.show()




