import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
from config import *
import csv

def eval_tree_diversity(compound_list, labels):
    diversity = []
    compound_list = np.array(compound_list)
    for l in list(set(labels)):
        all_compounds_for_l = compound_list[[i for i, val in enumerate(labels) if val==l]]
        #print(l, all_compounds__for_l)
        all_actions_for_i = []
        for c in all_compounds_for_l:
            all_actions_for_i.append(c.split("_")[1])
        #print(all_actions_for_i)
        diversity.append(len(list(set(all_actions_for_i))))
    return np.average(diversity)

def prediction_effects(compound_list, labels, cloest_comp):

    label = labels[compound_list.index(cloest_comp)]
    #print(label, labels)
    compound_list = np.array(compound_list)
    all_compounds_for_l = compound_list[[i for i, val in enumerate(labels) if val == label]]


    final_effect = []
    for c in all_compounds_for_l:
        final_effect.append(c.split("_")[1])
    final_effect = list(set(final_effect))
    #print(final_effect)
    return final_effect

def load_trained_model(link_path, train_data_path, test_data_path, thre_intervel=100):
    link = pd.read_csv(link_path,  usecols=["0", "1", "2", "3"])
    max_eucli_dis = np.max(link["2"])
    link = link.to_numpy()

    train_data = pd.read_csv(train_data_path, usecols=["Compound", "$t_l$", "$c_m$", "$c_{pt}$", "$t_r$", "$d_m$", "Action"])

    train_data['Compound'] += "_" + train_data['Action']
    train_data.set_index(['Compound'], drop=True, inplace=True)
    train_data = train_data.drop('Action', axis=1)
    train_data.index.name = None

    test_data = pd.read_csv(test_data_path, usecols=["Compound", "$t_l$", "$c_m$", "$c_{pt}$", "$t_r$", "$d_m$", "Action"])
    test_labels = test_data['Action']
    test_names = test_data['Compound']

    # normalize
    for f_name in train_data.columns[1:-1]:
        max_value = train_data[f_name].max()
        min_value = train_data[f_name].min()
        train_data[f_name] = (train_data[f_name] - min_value) / (max_value - min_value)
        train_data[f_name] = train_data[f_name] * 2 - 1

        test_data[f_name] = (test_data[f_name] - min_value) / (max_value - min_value)
        test_data[f_name] = test_data[f_name] * 2 - 1

    test_data['Compound'] += "_" + test_data['Action']
    test_data.set_index(['Compound'], drop=True, inplace=True)
    test_data = test_data.drop('Action', axis=1)
    test_data.index.name = None

    #test_data.set_index(['Compound'], drop=True, inplace=True)
    #test_data = test_data.drop('Action', axis=1)
    #test_data.index.name = None

    eval_metrics = []
    eval_metrics.append(["thre", "diversity", "error"])
    for thre in range(1,thre_intervel+1):
        t = thre/thre_intervel*max_eucli_dis

        #print(new_data, new_data.index)
        B = dendrogram(link, labels=list(train_data.index), color_threshold=t)
        #print(B.keys())
        #print(B['leaves_color_list'], B['ivl']) # NOTE  B['ivl']: the ordered leaves,  B['leaves_color_list']: labels
        diversity_h_cluster = eval_tree_diversity(compound_list=B['ivl'], labels=B['leaves_color_list'])
        #print(diversity_h_cluster)
        total_prediction = 0
        correct_pre = 0
        for index, test_d in test_data.T.iteritems():
            label = index.split("_")[1]
            dis = (train_data - list(test_d)).pow(2).sum(1).pow(0.5)
            cloest_comp = dis.idxmin()
            preds = prediction_effects(compound_list=B['ivl'], labels=B['leaves_color_list'], cloest_comp=cloest_comp)
            if label in preds:
                correct_pre += 1
                print(label, preds)
            total_prediction += 1
            #predictions.append(preds)

        accuracy = correct_pre / total_prediction
        print(thre/thre_intervel, diversity_h_cluster, 1-accuracy)
        eval_metrics.append([thre/thre_intervel, diversity_h_cluster, 1-accuracy])
        """
        for i in test_names:
            t_d = np.array(test_data.loc[i])
            data_t_dist = pdist(data_t)
            break
        """
    with open(RESULT_PATH / "hts_touch_response_eval_metrics.csv", "w") as save_csv:
        csv_writer = csv.writer(save_csv)
        csv_writer.writerows(eval_metrics)



if __name__ == "__main__":
    #save_binary_code_mapping_motion(
    #    binary_path="/Users/yankeewann/Desktop/HTScreening/data/featured/effects_binary_codes_with_integration.csv",
    #    save_path="/Users/yankeewann/Desktop/HTScreening/data/")

    #comp_names, data = read_binary_code_patterns(SAVE_FEATURE_PATH / ("effects_binary_codes_with_integration" + str(p_thre)+".csv"), DATA_PATH)
    load_trained_model(link_path=RESULT_PATH / "hierarchical_clustering_with_touch_response_linkage.csv",
                     train_data_path=DATASET_PATH/'distance_train.csv',
                     test_data_path=DATASET_PATH/'distance_test.csv',
                     thre_intervel=100)
