import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.patches as mpatches

from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

base_path = ["./data/behaviors_result07291500body/",
             "./data/behaviors_result07301500normalbody/",
             "./data/behaviors_result07311500body/"]

touching_part = ["body/"]
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
head_data = []
head_std = []
head_mean = []

body_data = []
body_std = []
body_mean = []

tail_data = []
tail_std = []
tail_mean = []

for I in range(4):
    paths = []
    for part in touching_part:
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

    """
    for i in range(5):
        concat_data = data_all[i+10] #data_all[i]+data_all[i+5]+
        concat_data = outlier_detection(concat_data)
        max_values.append(np.max(concat_data))
        head_data.append(concat_data)
        print("average for head: " + str(i+1), np.median(concat_data))
        head_std.append(np.std(concat_data))
        head_mean.append(np.mean(concat_data))
    """

    for i in range(3):
        concat_data = data_all[i]  # data_all[15+i]+data_all[15+i+5]+
        concat_data = outlier_detection(concat_data)
        max_values.append(np.max(concat_data))
        body_data.append(concat_data)
        print("average for body: " + str(i + 1), np.median(concat_data))
        body_std.append(np.std(concat_data))
        body_mean.append(np.mean(concat_data))

    """
    for i in range(5):
        concat_data = data_all[30+i+10] #data_all[30+i]+
        concat_data = outlier_detection(concat_data)
        max_values.append(np.max(concat_data))
        tail_data.append(concat_data)
        print("average for tail: " + str(i+1), np.median(concat_data))
        tail_std.append(np.std(concat_data))
        tail_mean.append(np.mean(concat_data))
    """

    max_value = np.max(max_values)

green_diamond = dict(markerfacecolor='y', marker='D')

ylabels = ["Latency Times (s)",
           "C-Shape Radius Average (pixels)",
           "Response Time (s)",
           "Moving Distance (pixels)"]
titles = ["Result of Latency Time \nof the Larva in Different Ages With Body Touched",
          "Result of C-Shape Radius \nof the Larva in Different Ages With Body Touched",
          "Result of Response Time \nof the Larva in Different Ages With Body Touched",
          "Result of Moving Distance \nof the Larva in Different Ages With Body Touched"]

colors = [[0.1843, 0.3098, 0.3098],
          [0.6843, 0.7098, 0.3098],
          [0.1843, 0.7098, 0.3098],
          [0.1843, 0.7098, 0.6098],
          [0.1843, 0.3098, 0.6098]]
# ========================================
x = [1, 2, 3, 4, 5]
# labels = ['30 hpf', '33 hpf', '51 hpf', '54 hpf', '57 hpf']
# boxes1 = plt.boxplot(head_data, labels = labels, positions = [1, 2, 3, 4, 5], widths = 0.8, patch_artist=True, showfliers=True)
# plt.errorbar(labels, head_mean,yerr=head_std, fmt='k-o',lw = 2,ecolor='k',elinewidth=1,ms=7,capsize=3)
# for box, color in zip(boxes1['boxes'], colors):
#    box.set(facecolor = color )

# ========================================
labels = ['30 hpf', '54 hpf', '78 hpf']
# boxes2 = plt.boxplot(body_data, labels = labels, positions = [7, 8, 9, 10, 11], widths = 0.8, patch_artist=True, showfliers=True)
# plt.errorbar(labels, body_mean, yerr=body_std, fmt='k-o',lw = 2,ecolor='k',elinewidth=1,ms=7,capsize=3)
# for box, color in zip(boxes2['boxes'], colors):
#    box.set(facecolor = color )
'''
plt.subplot(221)

plt.errorbar(labels, body_mean[0:3], yerr=body_std[0:3],  marker="^", lw=2, c='blue', elinewidth=1, ms=7, capsize=3)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[0], fontsize=8)
plt.title(ylabels[0], fontsize=12)

plt.subplot(222)

plt.errorbar(labels, body_mean[3:6], yerr=body_std[3:6], marker="^", lw=2, c='blue', elinewidth=1, ms=7, capsize=3)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[0], fontsize=8)
plt.title(ylabels[1], fontsize=12)

plt.subplot(223)

plt.errorbar(labels, body_mean[6:9], yerr=body_std[6:9], marker="^", lw=2, c='blue', elinewidth=1, ms=7, capsize=3)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[0], fontsize=8)
plt.title(ylabels[2], fontsize=12)


plt.subplot(224)
'''
plt.errorbar(labels, body_mean[9:12], yerr=body_std[9:12], marker="^", lw=2, c='blue', elinewidth=1, ms=7, capsize=3)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[0], fontsize=8)
plt.title(ylabels[3], fontsize=12)

# ========================================
# labels = ['30 hpf', '33 hpf', '51 hpf', '54 hpf', '57 hpf']
# boxes3 = plt.boxplot(tail_data, labels = labels, positions = [13, 14, 15, 16, 17], widths = 0.8, patch_artist=True, showfliers=True)
# plt.errorbar(labels, tail_mean, yerr=body_std, fmt='k-o',lw = 2,ecolor='k',elinewidth=1,ms=7,capsize=3)
# for box, color in zip(boxes3['boxes'], colors):
#    box.set(facecolor = color )

# plt.vlines(6, 0.001, max_value, linestyles='solid', colors='gray', alpha=0.2)
# plt.vlines(12, 0.001, max_value, linestyles='solid', colors='gray', alpha=0.2)

# patch1 = mpatches.Patch(color=[0.1843, 0.3098, 0.3098], label='30 hpf')
# patch2 = mpatches.Patch(color=[0.6843, 0.7098, 0.3098], label='33 hpf')
# patch3 = mpatches.Patch(color=[0.1843, 0.73098, 0.3098], label='51 hpf')
# patch4 = mpatches.Patch(color=[0.1843, 0.7098, 0.6098], label='54 hpf')
# patch5 = mpatches.Patch(color=[0.1843, 0.3098, 0.6098], label='57 hpf')

# plt.legend(handles=[patch1, patch2, patch3, patch4, patch5],loc = "best")
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
plt.show()

