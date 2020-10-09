import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.patches as mpatches

from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

base_path = ["./data_allthresholds/"]

comparison_path = ["behaviors_result07301530normalbody0.30/body/",
                 "behaviors_result07301530treatedbody0.30/body/"]
# all paths:
quatification = ["Latency_time.txt",
                 "C_Shape_radius.txt",
                 "Response_time.txt",
                 "Moving_distance.txt"]


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


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
add_zero = False

max_values = []
normal_data = []
treated_data = []

for I in range(4):
    paths = []
    for part in comparison_path:
        for base in base_path:
            paths.append(base + part + quatification[I])

    data_all = []

    for path in paths:
        data_file = open(path, 'r')
        data = data_file.readlines()
        distance_data = []
        for datum in data:
            data_item = (datum.strip())
            if data_item != "nan":
                distance_data.append((float)(data_item))
        # print(distance_data)
        # distance_data = np.array(distance_data)
        # inlierIndex = outlier_detection(distance_data)
        # distance_data = distance_data[inlierIndex]
        # distance_data = distance_data[np.where(distance_data<1)]
        # distance_data = extreme_renove(distance_data)
        data_all.append(distance_data)

    concat_data = data_all[0]  # +data_all[2]
    concat_data = outlier_detection(concat_data)
    normal_data.append(concat_data)
    print("average for head: ", np.median(concat_data))

    concat_data = data_all[1]  # +data_all[5]
    # concat_data = outlier_detection(concat_data)
    treated_data.append(concat_data)
    print("average for body: ", np.median(concat_data))

green_diamond = dict(markerfacecolor='y', marker='D')

ylabels = ["Latency Time (s)",
           "C-Bend Radius Average (pixels)",
           "Response Time (s)",
           "Moving Distance (pixels)"]
titles = ["Result of Latency Time \nof the Larva with Different Parts touched",
          "Result of C-Bend Radius \nof the Larva with Different Parts touched",
          "Result of Response Time \nof the Larva with Different Parts touched",
          "Result of Moving Distance \nof the Larva with Different Parts touched"]

colors = [[0.7843, 0.3098, 0.3098],
          [0.6843, 0.7098, 0.3098]]
# ========================================
plot_data = []
plot_data.append(normal_data)
plot_data.append(treated_data)
plot_data = np.array(plot_data)
labels = [['pos'], ['neg']]

plt.subplot(221)
boxes1 = plt.boxplot([normal_data[0], treated_data[0]],
                     labels=['pos', 'neg'], widths=0.3, positions=[1, 2], patch_artist=True,
                     showfliers=True, showmeans=True)
for box, color in zip(boxes1['boxes'], colors):
    box.set(facecolor=color)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[0], fontsize=8)
plt.title(ylabels[0], fontsize=12)

plt.subplot(222)
boxes2 = plt.boxplot([normal_data[1], treated_data[1]],
                     labels=['pos', 'neg'], widths=0.3, positions=[1, 2], patch_artist=True,
                     showfliers=True, showmeans=True)
for box, color in zip(boxes2['boxes'], colors):
    box.set(facecolor=color)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[1], fontsize=8)
# plt.title(titles[1])
plt.title(ylabels[1], fontsize=12)

plt.subplot(223)
boxes3 = plt.boxplot([normal_data[2], treated_data[2]],
                     labels=['pos', 'neg'], widths=0.3, positions=[1, 2], patch_artist=True,
                     showfliers=True, showmeans=True)
for box, color in zip(boxes3['boxes'], colors):
    box.set(facecolor=color)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[2], fontsize=8)
# plt.title(titles[2])
plt.title(ylabels[2], fontsize=12)

plt.subplot(224)

boxes3 = plt.boxplot([normal_data[3], treated_data[3]],
                     labels=['pos', 'neg'], widths=0.3, positions=[1, 2], patch_artist=True,
                     showfliers=True, showmeans=True)
for box, color in zip(boxes3['boxes'], colors):
    box.set(facecolor=color)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[3], fontsize=8)
# plt.title(titles[3])
plt.title(ylabels[3], fontsize=12)

patch1 = mpatches.Patch(color=[0.7843, 0.3098, 0.3098], label='pos')
patch2 = mpatches.Patch(color=[0.6843, 0.7098, 0.3098], label='neg')

# plt.legend(handles=[patch1, patch2, patch3],loc = "upper right")


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


plt.show()

