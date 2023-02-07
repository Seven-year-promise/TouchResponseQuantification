import pandas as pd
from config import *
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame



data = pd.read_csv(RESULT_PATH/"hts_touch_response_eval_metrics.csv", usecols=["thre", "diversity", "error"])

fig, ax = plt.subplots(figsize=(20,10))

data.plot.scatter(x = 'thre', y = 'diversity', ax = ax)
data.plot.scatter(x = 'thre', y = 'error', ax = ax, secondary_y = True)
plt.show()