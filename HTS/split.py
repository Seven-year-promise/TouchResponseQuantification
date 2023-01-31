import pandas as pd
from config import *


data = pd.read_csv(DATASET_PATH/'distance.csv', usecols=["Compound", "$t_l$", "$c_m$", "$c_{pt}$", "$t_r$", "$d_m$", "Action"])
test_data = data[data["Compound"].isin(TEST_COMPOUNDS)]
train_data = data[~data["Compound"].isin(TEST_COMPOUNDS)]

test_data.set_index(['Compound'], drop=True, inplace=True)
train_data.set_index(['Compound'], drop=True, inplace=True)

train_data.to_csv(DATASET_PATH/'distance_train.csv')
test_data.to_csv(DATASET_PATH/'distance_test.csv')