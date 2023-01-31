import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.patches as mpatches

from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

p_I = 0

base_path = ["./data/behaviors_result07301500normal_all/"]

touching_part = ["head/",
                 "body/",
                 "tail/"]
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
body_data = []
tail_data = []

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

    concat_data = data_all[0]  # + data_all[2] #+data_all[2]
    concat_data = outlier_detection(concat_data)
    max_values.append(np.max(concat_data))
    head_data.append(concat_data)
    print("average for head: ", np.median(concat_data))

    concat_data = data_all[1]  # + data_all[5] #+data_all[5]
    concat_data = outlier_detection(concat_data)
    max_values.append(np.max(concat_data))
    body_data.append(concat_data)
    print("average for body: ", np.median(concat_data))

    concat_data = data_all[2]  # + data_all[8] #+data_all[8]
    concat_data = outlier_detection(concat_data)
    max_values.append(np.max(concat_data))
    tail_data.append(concat_data)
    print("average for tail: ", np.median(concat_data))

    max_value = np.max(max_values)

green_diamond = dict(markerfacecolor='y', marker='D')
"""
ylabels = ["Latency time (s)",
           "C-Bend radius average (pixels)",
           "Response time (s)",
           "Moving distance (pixels)"]
"""
ylabels = ["$t_l$ (s)",
           "$r_a$ (pixels)",
           "$t_r$ (s)",
           "$d_m$ (pixels)"]
titles = ["Result of Latency Time \nof the Larva with Different Parts touched",
          "Result of C-Bend Radius \nof the Larva with Different Parts touched",
          "Result of Response Time \nof the Larva with Different Parts touched",
          "Result of Moving Distance \nof the Larva with Different Parts touched"]

colors = ['red',
          'blue',
          'green']
# ========================================
plot_data = []
plot_data.append(head_data)
plot_data.append(body_data)
plot_data.append(tail_data)
plot_data = np.array(plot_data)
labels = [['head'], ['body'], ['tail']]
plt.figure(figsize=(5,4))

#plt.subplot(221)
"""
boxes1 = plt.boxplot([head_data[0].tolist(), body_data[0].tolist(), tail_data[0].tolist()],
                     labels=['Head', 'Body', 'Tail'], widths=0.5, positions=[1, 2, 3], patch_artist=True,
                     showfliers=True, showmeans=True)
for box, color in zip(boxes1['boxes'], colors):
    box.set(facecolor=color)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[0], fontsize=8)
plt.xticks(fontsize=14, fontname = "Times New Roman")
plt.yticks(fontsize=14, fontname = "Times New Roman")
plt.title(ylabels[0], fontname = "Times New Roman", fontsize=14)


plt.subplot(222)

boxes2 = plt.boxplot([head_data[1].tolist(), body_data[1].tolist(), tail_data[1].tolist()],
                     labels=['Head', 'Body', 'Tail'], widths=0.5, positions=[4, 5, 6], patch_artist=True,
                     showfliers=True, showmeans=True)
for box, color in zip(boxes2['boxes'], colors):
    box.set(facecolor=color)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[1], fontsize=8)
# plt.title(titles[1])
plt.xticks(fontsize=14, fontname = "Times New Roman")
plt.yticks(fontsize=14, fontname = "Times New Roman")
plt.title(ylabels[1], fontname = "Times New Roman", fontsize=14)

plt.subplot(223)
"""
boxes3 = plt.boxplot([head_data[p_I].tolist(), body_data[p_I].tolist(), tail_data[p_I].tolist()],
                     labels=['Head', 'Body', 'Tail'], widths=0.5, positions=[7, 8, 9], patch_artist=True,
                     showfliers=True, showmeans=True)
for box, color in zip(boxes3['boxes'], colors):
    box.set(facecolor=color)
plt.grid(b=True, which="both", axis="both")
# plt.ylabel(ylabels[2], fontsize=8)
# plt.title(titles[2])
plt.xticks(fontsize=14, fontname = "Arial")
plt.yticks(fontsize=14, fontname = "Arial")
plt.ylabel(ylabels[p_I], fontname = "Arial", fontsize=14)
"""
plt.subplot(224)


boxes3 = plt.boxplot([head_data[3].tolist(), body_data[3].tolist(), tail_data[3].tolist()],
                     labels=['Head', 'Body', 'Tail'], widths=0.5, positions=[7, 8, 9], patch_artist=True,
                     showfliers=True, showmeans=True)
for box, color in zip(boxes3['boxes'], colors):
    box.set(facecolor=color)
plt.grid(b=True, which="both", axis="both")
plt.xticks(fontsize=14, fontname = "Times New Roman")
plt.yticks(fontsize=14, fontname = "Times New Roman")
plt.title(ylabels[3], fontname = "Times New Roman", fontsize=14)
"""

patch1 = mpatches.Patch(color=[0.7843, 0.3098, 0.3098], label='head')
patch2 = mpatches.Patch(color=[0.6843, 0.7098, 0.3098], label='body')
patch3 = mpatches.Patch(color=[0.1843, 0.23098, 0.7098], label='tail')

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

plt.grid(b=None)
plt.tight_layout()
plt.show()

