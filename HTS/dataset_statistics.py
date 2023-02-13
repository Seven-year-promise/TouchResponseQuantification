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
    num_train = 0
    num_test = 0
    for c_name in data.keys():
        if c_name in TEST_COMPOUNDS:
            num_test += len(data[c_name])
        else:
            num_train += len(data[c_name])
    print("num of train data: ", num_train)
    print("num of test data: ", num_test)
if __name__ == "__main__":
    quan_data = read_data(QUANTIFY_DATA_PATH)
    action_mode = read_actions(ACTION_DATA_PATH)
    distance_data = compute_distance(quan_data, action_mode)