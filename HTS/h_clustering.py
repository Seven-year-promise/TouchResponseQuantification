from config import *
from distribution_distance import compute_prob_distance
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import *
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering

def hierarchical_clustering(data):
    data['Compound'] += "_" + data['Action']

    #normalize
    for f_name in data.columns[1:-1]:
        print(f_name)
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

def extract_clustered_table(res, data):
    """
    input
    =====
    res:       the clustermap object
    data:                input table

    output
    ======
    returns:             reordered input table
    """

    # if sns.clustermap is run with row_cluster=False:
    if res.dendrogram_row is None:
        print("Apparently, rows were not clustered.")
        return -1

    if res.dendrogram_col is not None:
        # reordering index and columns
        new_cols = data.columns[res.dendrogram_col.reordered_ind]
        new_ind = data.index[res.dendrogram_row.reordered_ind]

        return data.loc[new_ind, new_cols]

    else:
        # reordering the index
        new_ind = data.index[res.dendrogram_row.reordered_ind]

        return data.loc[new_ind, :]

def HI_clustering_sns(path):
    data = pd.read_csv(path, usecols=["Compound", "$t_l$", "$c_m$", "$c_{pt}$", "$t_r$", "$d_m$", "Action"])
    data['Compound'] += "_" + data['Action']

    # normalize
    for f_name in data.columns[1:-1]:
        print(f_name)
        max_value = data[f_name].max()
        min_value = data[f_name].min()
        data[f_name] = (data[f_name] - min_value) / (max_value - min_value)
        data[f_name] = data[f_name] * 2 - 1

    data.set_index(['Compound'], drop=True, inplace=True)
    data = data.drop('Action', axis=1)
    data.index.name = None

    #fig = plt.figure(figsize=(15, 20))
    heatmap = sns.clustermap(data=data, method='ward', metric='euclidean',
                             row_cluster=True, col_cluster=None, cmap="coolwarm",
                             vmin=-1, vmax=1, figsize=(15, 55))

    heatmap.fig.suptitle("Hierarchy Clustering", fontsize=20)
    heatmap.ax_heatmap.set_title("Hierarchical Clustering(ward)", fontsize=20)
    # plt.show()
    plt.setp(heatmap.ax_heatmap.get_yticklabels(), rotation=0)
    plt.savefig(RESULT_PATH / (
                "hierarchical_clustering_with_touch_response_heatmap.png"), dpi=300)
    plt.clf()


    ordered_data = extract_clustered_table(heatmap, data)
    ordered_data.to_csv(RESULT_PATH / "hierarchical_clustering_with_touch_response_ordered_cluster.csv")

    linkage = pd.DataFrame(heatmap.dendrogram_row.linkage)
    linkage.to_csv(RESULT_PATH / "hierarchical_clustering_with_touch_response_linkage.csv")

    B = dendrogram(linkage, labels=list(data.index), color_threshold=1)#, p=20, truncate_mode='level', color_threshold=1)
    plt.savefig(RESULT_PATH / "hierarchical_clustering_with_touch_response_tree.png", dpi=300)

    #print(heatmap.dendrogram_row.linkage)


    """
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

    model = model.fit(data)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    """

if __name__ == "__main__":
    HI_clustering_sns(path=DATASET_PATH/'distance_train.csv')