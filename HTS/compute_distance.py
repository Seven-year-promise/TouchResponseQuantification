from config import *
from distribution_distance import compute_prob_distance
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import *
import seaborn as sns

def read_data(path) -> dict:
    csv_files = list(path.rglob("*.csv"))
    all_data = {}
    #print(path)
    comp_name = ""
    for c_file in csv_files:
        with open(c_file, newline='') as c_f:
            c_f_reader = csv.reader(c_f, delimiter=';', quotechar='|')
            for i, row in enumerate(c_f_reader):
                if i > 1:
                    comp_name = row[0]
                    if comp_name not in all_data.keys():
                        all_data[comp_name] = []
                    if row[2] == "":
                        all_data[comp_name].append([15.0]+[float(x) for x in row[3:]])
                    elif row[2] != "-2":
                        all_data[comp_name].append([float(x) for x in row[2:]])
                    else:
                        pass
    return all_data

def compute_distance(data, action_mode):
    distance_features = []

    WILD_inte = np.array(data["C0"]).reshape(len(data["C0"]), 5)
    for c_name in data.keys():
        c_data = data[c_name]
        if c_name == "C15":
            print(c_data)
        if len(c_data)> 0:
            c_data = np.array(c_data).reshape(len(c_data), 5)
            c_action = get_key(action_mode, c_name)
            distance_data = [c_name]

            for ph in range(5):
                dis = compute_prob_distance(WILD_inte[:, ph], c_data[:, ph], algorithm="wasserstein_2")
                # print(dis)
                distance_data.append(dis)

            distance_data.append(c_action)
            distance_features.append(distance_data)

    return pd.DataFrame(distance_features, columns = ["Compound", "$t_l$", "$c_m$", "$c_{pt}$", "$t_r$", "$d_m$", "Action"])

def hierarchical_clustering(data):
    data['Compound'] += "_" + data['Action']

    #normalize
    for f_name in data.columns[1:-1]:
        #print(f_name)
        max_value = data[f_name].max()
        min_value = data[f_name].min()
        data[f_name] = (data[f_name] - min_value) / (max_value - min_value)
        data[f_name] = data[f_name]*2 - 1

    data.set_index(['Compound'], drop=True, inplace=True)
    data = data.drop('Action', axis=1)
    data.index.name = None

    # fig = plt.figure(figsize=(15, 20))
    heatmap = sns.clustermap(data=data, method='ward', metric='euclidean',
                             row_cluster=True, col_cluster=None, cmap="coolwarm",
                             vmin=-1, vmax=1, figsize=(15, 55))

    heatmap.fig.suptitle("Hierarchy Clustering", fontsize=20)
    heatmap.ax_heatmap.set_title("Hierarchical Clustering(ward)", fontsize=20)
    # plt.show()
    plt.setp(heatmap.ax_heatmap.get_yticklabels(), rotation=0)
    plt.savefig(RESULT_PATH / (
                "hierarchical_clustering_with_touch_response.png"),
                dpi=300)

def pattern_prediction(data):
    pass



if __name__ == "__main__":
    quan_data = read_data(QUANTIFY_DATA_PATH)
    action_mode = read_actions(ACTION_DATA_PATH)
    distance_data = compute_distance(quan_data, action_mode)
    distance_data.set_index(['Compound'], drop=True, inplace=True)
    distance_data.to_csv(DATASET_PATH/'distance.csv')
    #hierarchical_clustering(distance_data)
    #print(distance_data)