from sklearn.cluster import KMeans
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

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
            group_weight = group_weight/np.sum(group_weight)
        edge_list.append(group_weight)

    return edge_list

def visualize_cluster_results(group_list):
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
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1]) 
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])             
        ax.set_zticks([0, 0.25, 0.5, 0.75, 1])      

        # ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], c='red', marker='x', label='Cluster Centers')
        # plt.legend()
        # plt.show()
        plt.savefig("cluster_results/group"+str(i+1)+".png")


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

# visualize_cluster_results(group_list)
print(cluster_labels)
print(centers)
# figure out which cluster is leader and supporter

leader_ratio_list = []

for i in range(len(cluster_labels)):
    label_1 = np.count_nonzero(cluster_labels[i])
    label_0 = 4 - label_1
    count = 0
    if centers[i][0][0] > centers[i][1][0]:
        count = count + 1
    if centers[i][0][1] > centers[i][1][1]:
        count = count + 1
    if centers[i][0][2] > centers[i][1][2]:
        count = count + 1

    if count >= 2: # label 0 is leader
        leader_ratio_list.append(label_0/4)
    else: # label 1 is leader
        leader_ratio_list.append(label_1/4)

df = pd.read_csv('completion_time_and_accuracy.csv')
data = df.to_numpy()
accuracy = np.array(df["Accuracy (%)"])
complete_time = []
for i in range(len(df["Completion Time"])):
    time = df["Completion Time"][i]
    match_min = re.search(r'\d{2}:(\d{2}):\d{2}\.\d+', time)
    match_sec = re.search(r'\d{2}:\d{2}:(\d{2}\.\d+)', time)
    total_sec = int(match_min.group(1)) * 60 + eval(match_sec.group(1))
    complete_time.append(total_sec)
complete_time = np.array(complete_time)
effciency = []

for i in range(len(complete_time)):
    effciency.append(accuracy[i]/complete_time[i])

print(complete_time)
print(accuracy)
print(leader_ratio_list)

plt.figure(1)
plt.scatter(leader_ratio_list, accuracy)
plt.xticks([0,0.25,0.5,0.75,1])
plt.xlabel("leader ratio")
plt.ylabel("accuracy(%)")
plt.savefig("accuracy.png")

plt.figure(2)
plt.scatter(leader_ratio_list, complete_time)
plt.xticks([0,0.25,0.5,0.75,1])
plt.xlabel("leader ratio")
plt.ylabel("complete time(s)")
plt.savefig("time.png")

plt.figure(3)
plt.scatter(leader_ratio_list, effciency)
plt.xticks([0,0.25,0.5,0.75,1])
plt.xlabel("leader ratio")
plt.ylabel("effciency (accuracy/time)")
plt.savefig("effciency.png")
# plt.show()