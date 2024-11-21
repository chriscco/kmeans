from sklearn.cluster import KMeans
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_group_number(graph_id):
    match = re.search(r'group-(\d+)', graph_id)
    return int(match.group(1)) if match else float('inf')  # If no group number, place it at the end

# get weight for each node: A,B,C,D
def get_weight(json_file, normalize = False):
    with open(json_file, 'r') as f:
        pro_data = json.load(f)

    sorted_graphs = sorted(pro_data["graphs"], key=lambda x: extract_group_number(x["id"]))
    edge_list = []

    node_index = {"A":0, "B":1, "C":2, "D":3}

    for i, graph_data in enumerate(sorted_graphs):
        group_weight= [0,0,0,0] # A,B,C,D
        for edge in graph_data["edges"]:
            source = edge["source"]
            target = edge["target"]
            weight = edge["metadata"]["weight"]
            group_weight[node_index[source]] = round(group_weight[node_index[source]] + weight,3)
            group_weight[node_index[target]] = round(group_weight[node_index[target]] + weight,3)
        if normalize:
            group_weight = group_weight/np.max(group_weight)
        edge_list.append(group_weight)

    return edge_list



json_file_list = ["proximity_graphs.json","conversation_graphs.json", "shared_attention_graphs.json"]
proximity_weight_list = get_weight(json_file_list[0])
conversation_weight_list = get_weight(json_file_list[1])
attention_weight_list = get_weight(json_file_list[2])

normal_proximity_weight_list = get_weight(json_file_list[0], normalize=True)
normal_conversation_weight_list = get_weight(json_file_list[1], normalize=True)
normal_attention_weight_list = get_weight(json_file_list[2], normalize=True)

group_list = []
for i in range(len(proximity_weight_list)):
    group = []
    for j in range(len(proximity_weight_list[0])):
        group.append([normal_proximity_weight_list[i][j], normal_conversation_weight_list[i][j], normal_attention_weight_list[i][j]])
    group_list.append(group)

cluster_labels = []
centers = []
inertia = []
for group in group_list:
    x = np.array(group)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    kmeans.fit(x)
    cluster_labels.append(kmeans.labels_)
    centers.append(kmeans.cluster_centers_)
    inertia.append(kmeans.inertia_)

for i in range(len(group_list)):
    X = np.array(group_list[i])

    colors = ['red', 'blue']
    mapped_colors = [colors[label] for label in cluster_labels[i]]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=mapped_colors, label='Data Points')
    ax.set_xlim(np.min(X[:, 0]),np.max(X[:, 0]))
    ax.set_ylim(np.min(X[:, 1]),np.max(X[:, 1]))
    ax.set_zlim(np.min(X[:, 2]),np.max(X[:, 2]))

    ax.set_zlabel('proximity', fontdict = {"size":15, 'color': 'red'})
    ax.set_ylabel('conversation', fontdict = {"size":15, 'color': 'red'})
    ax.set_xlabel('attention', fontdict = {"size":15, 'color': 'red'})
    # ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], c='red', marker='x', label='Cluster Centers')
    # plt.legend()
    # plt.show()
    plt.savefig("cluster_results/group"+str(i+1)+".png")