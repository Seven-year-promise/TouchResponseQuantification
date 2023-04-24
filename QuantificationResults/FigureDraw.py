import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

base_path = ["./20210522-4compounds/used_CI/Control/",
             "./20210522-4compounds/used_CI/DMSO/",
             "./20210522-4compounds/used_CI/Dia/",
             "./20210522-4compounds/used_CI/Iso/",
             "./20210522-4compounds/used_CI/Caffine/",
             "./20210522-4compounds/used_CI/Saha/"]
# all paths:
quatification = "quantification.csv"
index = ["t_l", "c_m", "cpt", "t_r", "d_m"]
I = 4

def outlier_detection(X):
    X = np.array(X)
    elements = X.reshape(-1, 1)

    # detector = LocalOutlierFactor(n_neighbors=10, algorithm = 'kd_tree')

    # return np.where(detector.fit_predict(X)==1)

    # detector = EllipticEnvelope(random_state=0).fit(elements)
    detector = IsolationForest(random_state=0).fit(elements)

    # predict returns 1 for an inlier and -1 for an outlier
    prediction = np.where(detector.predict(elements) == 1)
    final_list = X[prediction[0]]
    # y = np.arange(len(final_list))
    # plt.scatter(y, final_list)
    # plt.show()
    return final_list


def extreme_renove(X):
    elements = np.array(X)
    # y = np.arange(len(X))
    # plt.scatter(y, X)
    # plt.show()

    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)

    final_list = [x for x in X if (x > mean - 2 * sd)]
    final_list = [x for x in final_list if (x < mean + 2 * sd)]
    """
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))

    def Gauss(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    popt,pcov = curve_fit(Gauss, x, y, p0=[max(y), mean, sigma])

    """
    # y = np.arange(len(final_list))
    # plt.scatter(y, final_list)
    # plt.show()

    return final_list

def csv_reader(path, head):
    data = pd.read_csv(path, sep=',')
    data = data.dropna(subset=[head])
    readout = data[head]
    return readout

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
add_zero = False

max_values = []
stds = []
means = []


data_all = []

for path in base_path:
    data = csv_reader(path + quatification, index[I])
    print(data)
    clean_data = []
    for datum in data:
        datum = float(datum)
        print(datum)
        if datum != -2:
            clean_data.append((float)(datum))
    # print(distance_data)
    # distance_data = np.array(distance_data)
    # inlierIndex = outlier_detection(distance_data)
    # distance_data = distance_data[inlierIndex]
    # distance_data = distance_data[np.where(distance_data<1)]
    # distance_data = extreme_renove(distance_data)
    filter_data = outlier_detection(clean_data)
    max_values.append(np.max(filter_data))
    stds.append(np.std(filter_data))
    means.append(np.mean(filter_data))
    data_all.append(filter_data)


green_diamond = dict(markerfacecolor='y', marker='D')
"""
ylabels = ["Latency Times (s)",
           "C-Bend Curvature Maximum",
           "C-Bend Curvature Peak Time",
           "Response Time (s)",
           "Moving Distance (pixels)"]
"""
ylabels = ["$t_l (s)$",
           "$c_m$",
           "$t_{cp} (s)$",
           "$t_r (s)$",
           "$d_m (pixels)$"]
titles = ["Result of Latency Time \nof the Larvae in Different Ages With Body Touched",
          "Result of C-Bend Radius \nof the Larva in Different Ages With Body Touched",
          "Result of Response Time \nof the Larva in Different Ages With Body Touched",
          "Result of Moving Distance \nof the Larva in Different Ages With Body Touched"]

colors = [[0.1843, 0.3098, 0.3098],
          [0.6843, 0.7098, 0.5098],
          [0.1843, 0.7098, 0.3098],
          [0.1843, 0.2098, 0.9098],
          [0.1843, 0.3098, 0.4098]]
colors = ["tab:blue", "tab:green",  "tab:purple", "tab:brown", "tab:pink", "tab:olive"]
# ========================================
x = [1, 2, 3, 4, 5]
labels = ["$Wild$", "$DMSO$", "$Dia$", "$Iso$", "$Caffi$", "$Saha$"]
boxes1 = plt.boxplot(data_all, labels = labels, positions = [0, 1, 2, 3, 4, 5], widths = 0.8, patch_artist=True,
                     showfliers=True)
#plt.errorbar(labels, means,yerr=stds, fmt='k-o',lw = 2,ecolor='k',elinewidth=1,ms=7,capsize=3)
for box, color in zip(boxes1['boxes'], colors):
    box.set(facecolor = color)

plt.ylabel(ylabels[I], fontsize=20, fontname = "Arial")
plt.xticks(fontsize=20, fontname = "Arial")
plt.yticks(fontsize=20, fontname = "Arial")
#plt.yticks(fontsize=12, fontname = "Times New Roman")
# ========================================
#labels = ['54', '54.5', '55', '55.5', '56', '56.5']
# boxes2 = plt.boxplot(body_data, labels = labels, positions = [7, 8, 9, 10, 11], widths = 0.8, patch_artist=True, showfliers=True)
# plt.errorbar(labels, body_mean, yerr=body_std, fmt='k-o',lw = 2,ecolor='k',elinewidth=1,ms=7,capsize=3)
# for box, color in zip(boxes2['boxes'], colors):
#    box.set(facecolor = color )
#plt.figure(figsize=(4,4))
#plt.subplot(221)
"""
plt.errorbar(labels, normal_mean[0:6], yerr=normal_std[0:6], lw=2, marker="^", c='blue', elinewidth=1, ms=7, capsize=3)
plt.errorbar(labels, treated_mean[0:6], yerr=treated_std[0:6], marker="o", lw=2, c='pink', elinewidth=1, ms=7,
             capsize=3)
plt.grid(b=True, which="both", axis="both")
#plt.ylabel(ylabels[0], fontsize=12)
plt.xticks(fontsize=14, fontname = "Times New Roman")
plt.yticks(fontsize=14, fontname = "Times New Roman")
plt.title(ylabels[0], fontname = "Times New Roman", fontsize=14)


plt.subplot(222)

plt.errorbar(labels, normal_mean[6:12], yerr=normal_std[6:12], lw=2, marker="^", c='blue', elinewidth=1, ms=7, capsize=3)
plt.errorbar(labels, treated_mean[6:12], yerr=treated_std[6:12], marker="o", lw=2, c='pink', elinewidth=1, ms=7,
             capsize=3)
plt.grid(b=True, which="both", axis="both")
#plt.ylabel(ylabels[1], fontsize=12)
plt.xticks(fontsize=14, fontname = "Times New Roman")
plt.yticks(fontsize=14, fontname = "Times New Roman")
plt.title(ylabels[1], fontname = "Times New Roman", fontsize=14)

plt.subplot(223)

plt.errorbar(labels, normal_mean[12:18], yerr=normal_std[12:18], lw=2, marker="^", c='blue', elinewidth=1, ms=7,
             capsize=3)
plt.errorbar(labels, treated_mean[12:18], yerr=treated_std[12:18], marker="o", lw=2, c='pink', elinewidth=1, ms=7,
             capsize=3)
plt.grid(b=True, which="both", axis="both")
#plt.ylabel(ylabels[2], fontsize=12)
plt.xticks(fontsize=14, fontname = "Times New Roman")
plt.yticks(fontsize=14, fontname = "Times New Roman")
plt.title(ylabels[2], fontname = "Times New Roman", fontsize=14)

plt.subplot(224)

plt.errorbar(labels, normal_mean[18:24], yerr=normal_std[18:24], lw=2, marker="^", c='blue', elinewidth=1, ms=7,
             capsize=3)
plt.errorbar(labels, treated_mean[18:24], yerr=treated_std[18:24], marker="o", lw=2, c='pink', elinewidth=1, ms=7,
             capsize=3)
plt.grid(b=True, which="both", axis="both")
#plt.ylabel(ylabels[3], fontsize=12)
plt.xticks(fontsize=14, fontname = "Times New Roman")
plt.yticks(fontsize=14, fontname = "Times New Roman")
plt.title(ylabels[3], fontname = "Times New Roman", fontsize=14)
"""

#patch1 = mpatches.Patch(color='blue', label='controls')
#patch2 = mpatches.Patch(color='pink', label='treated larvae')
#plt.legend(handles=[patch1, patch2],loc = "upper left")
# ========================================
# labels = ['30 hpf', '33 hpf', '51 hpf', '54 hpf', '57 hpf']
# boxes3 = plt.boxplot(tail_data, labels = labels, positions = [13, 14, 15, 16, 17], widths = 0.8, patch_artist=True, showfliers=True)
# plt.errorbar(labels, tail_mean, yerr=body_std, fmt='k-o',lw = 2,ecolor='k',elinewidth=1,ms=7,capsize=3)
# for box, color in zip(boxes3['boxes'], colors):
#    box.set(facecolor = color )

# plt.vlines(6, 0.001, max_value, linestyles='solid', colors='gray', alpha=0.2)
# plt.vlines(12, 0.001, max_value, linestyles='solid', colors='gray', alpha=0.2)

#patch1 = mpatches.Patch(color='blue', label='controls')
#patch2 = mpatches.Patch(color='pink', label='treated larvae')
#plt.legend(handles=[patch1, patch2],loc = "up_right")

"""
for i in range(len(df['class'].unique())-1):
    plt.vlines(i+.5, 10, 45, linestyles='solid', colors='gray', alpha=0.2)
"""

# box_1 = plt.boxplot(data1, label = 'head')
# box_2 = plt.boxplot(data2, label = 'body')
# box_3 = plt.boxplot(data3, label = 'tail')
# plt.legend(loc = 'upper right')
"""
colors = ['blue', 'red', 'green']
for i in [1,2,3]:
    y = data[i-1]
    # Add some random "jitter" to the x-axis
    x = np.random.normal(i, 0.04, size=len(y))
    plt.plot(x, y, 'r.', alpha=0.5, color = colors[i-1])
"""
# plt.xlabel("The fish part touched: 1 Head, 2 Body, 3 Tail")

# plt.ylabel(ylabels[I])

# plt.title(titles[I])
plt.tight_layout()
plt.show()

