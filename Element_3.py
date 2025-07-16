import math
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from vincenty import vincenty
from reportlab.pdfgen import canvas
from io import BytesIO
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg

# Load the node data from a CSV file
data = pd.read_csv('CC_locations.csv')

# Create a dictionary to store the node information
nodes = {}

# Loop over the rows of the DataFrame and add each node to the dictionary
for i, row in data.iterrows():
    node_id = row['City']
    x = row['X']
    y = row['Y']
    nodes[node_id] = (x, y)

# Specify the number of centroids for KMeans function to resolve
num_clusters = 2

# Assign each city to its nearest cluster center (centroid) which identifies the KMeans centre as optimal locations
# this provides two points to then plot a path between
cluster_centers = []
optimal_cities = []

# Create a KMeans object and fit the data to the model
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data[['X', 'Y']

for j in range(num_clusters):
    cities_in_cluster = data[kmeans.labels_ == j]
    if len(cities_in_cluster) > 0:
        nearest_city_index = ((cities_in_cluster['X'] - kmeans.cluster_centers_[j, 0]) ** 2 + (
                cities_in_cluster['Y'] - kmeans.cluster_centers_[j, 1]) ** 2).idxmin()
        nearest_city_location = cities_in_cluster.loc[nearest_city_index, ['X', 'Y']].to_numpy()
        optimal_cities.append(cities_in_cluster.loc[nearest_city_index, ['City'][0]])
        cluster_centers.append(nearest_city_location)

# Define the output of the above to variables Source and Target, for the 'dijkstra' function to utilise
source = optimal_cities[0]
target = optimal_cities[1]

# Define the required Edges as a dictionary
edges = [('A', 'Q'), ('A', 'V'), ('A', 'J'), ('B', 'E'), ('B', 'K'), ('B', 'Y'), ('C', 'O'),
         ('C', 'G'), ('C', 'S'), ('C', 'W'), ('D', 'I'), ('D', 'M'), ('E', 'T'), ('E', 'L'),
         ('F', 'R'), ('F', 'Z'), ('F', 'O'), ('G', 'P'), ('G', 'X'), ('H', 'I'), ('H', 'L'),
         ('H', 'R'), ('H', 'M'), ('I', 'N'), ('J', 'P'), ('J', 'K'), ('K', 'T'), ('L', 'U'),
         ('N', 'S'), ('P', 'X'), ('P', 'W'), ('S', 'W'), ('T', 'V'), ('X', 'Z'), ('Y', 'Z')]

# Create a List to store the Distance between each Edge connected Node
edge_len = []

# Iterate through Dictionary of Edges and calculate distance between each pair using Euclidean Distance
# Appending records to edge_len and create Adjacency Matrix from distance weights
num_nodes = len(nodes)
adj_matrix = np.zeros((num_nodes, num_nodes))
node_list = list(nodes.keys())
node_indices = {node_list[i]: i for i in range(num_nodes)}
for source, target in edges:
    source_index = node_indices[source]
    target_index = node_indices[target]
    x1 = nodes[source][0]
    y1 = nodes[source][1]
    x2 = nodes[target][0]
    y2 = nodes[target][1]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    edge_len.append((source, target, vincenty(nodes[source], nodes[target], miles=True)))
    adj_matrix[source_index][target_index] = distance
    adj_matrix[target_index][source_index] = distance


# Define the implementation of Dijkstra's algorithm using previously calculated 'optimal' cities and the weighted
# Adjacency Matrix of Edge lengths
def dijkstra(start, end, matrix):
    matrix_nodes = len(matrix)
    visited = [False] * matrix_nodes
    distances = [float('inf')] * matrix_nodes
    distances[node_indices[start]] = 0
    find_path = [-1] * matrix_nodes

    for o in range(matrix_nodes):
        min_distance = float('inf')
        min_index = -1
        for m in range(matrix_nodes):
            if not visited[m] and distances[m] < min_distance:
                min_distance = distances[m]
                min_index = m

        if min_index == -1:
            break

        visited[min_index] = True

        for n in range(matrix_nodes):
            if matrix[min_index][n] > 0 and not visited[n]:
                new_distance = distances[min_index] + matrix[min_index][n]
                if new_distance < distances[n]:
                    distances[n] = new_distance
                    find_path[n] = min_index

    shortest_path = []
    current_node = node_indices[end]
    while current_node != -1:
        shortest_path.append(node_list[current_node])
        current_node = find_path[current_node]
    shortest_path.reverse()

    return shortest_path


# Execute the Dijkstra function, returning the path taken between the optimal locations and calculate total
# distance in Miles
path = dijkstra(source, target, adj_matrix)
optimal_distance = 0.00

count = len(path) - 1
while count > 0:
    for j in edge_len:
        if path[count] in j and path[count - 1] in j:
            count = count - 1
            optimal_distance = optimal_distance + float(j[2])

# Text Strings to be provided alongside the visual representation of the Node, Edges and Optimal Path, providing context
optimal_distance = f"The optimal distance between {source} and {target} is {optimal_distance:.2f} Miles."
optimal_path = f"The optimal path is {path}."

# Define Coordinate References using Plate Carree Projection
projection = ccrs.PlateCarree()

# Create a figure and axis object
fig, ax = plt.subplots(subplot_kw=dict(projection=projection))
fig.set_figwidth(5)
fig.set_figheight(5)

# Define OpenStreet Maps for background layer
ax.add_feature(cfeat.COASTLINE)

extent = [(min(data['X'])) - .4, (max(data['X'])) + .4, (min(data['Y'])) - 1, (max(data['Y'])) + 1]
ax.set_extent(extent)

# Plot the node locations
for node, position in nodes.items():
    ax.scatter(position[0], position[1], color='r', zorder=4)

# Add labels to the nodes
for node, position in nodes.items():
    ax.text(position[0], position[1], node, fontsize=4, zorder=2)

# Plot 'optimal' city node identifiers
for i, center in enumerate(cluster_centers):
    city_name = data[kmeans.labels_ == i]['City'].values[0]
    ax.scatter(center[0], center[1], s=100, c='blue', marker='*', zorder=5)

# Plot each of the Edges
for (source, target) in edges:
    ax.plot([nodes[source][0], nodes[target][0]], [nodes[source][1], nodes[target][1]], '-', color='k', zorder=1)

# Plot length of each Edges central to the Edge
for source, target, length in edge_len:
    x1 = float(nodes[source][0])
    x2 = float(nodes[target][0])
    y1 = float(nodes[source][1])
    y2 = float(nodes[target][1])
    x_loc = (x1 + x2) / 2
    y_loc = (y1 + y2) / 2
    length = float(length)
    ax.annotate(f'{length:.2f}', (x_loc-.2, y_loc+.2), ha='center', fontsize=4, color='red', zorder=9, )

# Plot 'optimal' route between identified 'optimal' cities
for k in range(len(path) - 1):
    source = path[k]
    target = path[k + 1]
    ax.plot([nodes[source][0], nodes[target][0]], [nodes[source][1], nodes[target][1]], '-', linewidth=5, color='g',
            alpha=0.5, zorder=6)


# Define Title for Plot
ax.set_title('City Locations')

# Setup PDF Output
imgdata = BytesIO()
fig.savefig(imgdata, format='svg')
imgdata.seek(0)

drawing = svg2rlg(imgdata)

c = canvas.Canvas('Optimal Path.pdf')
renderPDF.draw(drawing, c, 70, 400)
c.drawString(50, 400, optimal_path)
c.drawString(50, 380, optimal_distance)
c.showPage()
c.save()
