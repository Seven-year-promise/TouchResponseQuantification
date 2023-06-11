import pandas as pd
from config import *
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame



data = pd.read_csv(RESULT_PATH/"hts_touch_response_eval_metrics.csv", usecols=["Threshold", "Diversity", "Failures"])

#thresholds = list(data['Threshold of hierarchical clustering'])
#diversity = list(data['Diversity of drug effect tree'])
#failure = list(data['Failures of prediction'])
#data.plot(x = 'Threshold of hierarchical clustering', y = r"Diversity of drug effect tree", ax = ax)
#data.plot(x = 'Threshold of hierarchical clustering', y = r'Failures of prediction', ax = ax, secondary_y = True)
fig,ax = plt.subplots()
# make a plot
ax.plot(data.Threshold,
        data.Diversity,
        color= "black",
        linestyle="--",
        label=r"#$N_{pp}$")
# set x-axis label
ax.set_xlabel(r"Threshold of hierarchical clustering ($T_{pr}$)", fontname = "Arial", fontsize = 12)
# set y-axis label
ax.set_ylabel(r"Number of pattern proposals (#$N_{pp}$)", #r"Diversity of drug effect tree ($#TD$)",
              fontname = "Arial",
              fontsize=12)

plt.scatter(x=0.3, y=4.4, color="red", s=20)
plt.text(x = 0.31, y = 4.1, color = 'r', s = '(0.3, 4.4)', fontname = "Arial", fontsize=10)



ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(data.Threshold, data.Failures, color="black", linestyle="-", label=r"$P_f$")
ax2.set_ylabel(r"Percentage of failures of pattern prediction ($P_f$)", fontname = "Arial", fontsize=12)


plt.axvline(x = 0.3, color = 'r', linestyle=":")

plt.scatter(x=0.3, y=0.1333, color="red", s=20)
plt.text(x = 0.13, y = 0.14, color = 'r', s = '(0.3, 0.13)', fontname = "Arial", fontsize=10)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="center right")
plt.tight_layout()

plt.savefig(RESULT_PATH / "prtd_eval.eps", dpi=300)
#plt.show()