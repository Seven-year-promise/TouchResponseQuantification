import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from sklearn.neighbors import KernelDensity
from scipy.stats import norm

font = "Calibri"
fontsize = 14

def hist_percentage(data, bin_num = 20):
    d_min = np.min(data)
    d_max = np.max(data)
    print(d_min, d_max)
    stride = (d_max-d_min)/(bin_num*1.0)
    total_num = len(data)
    hist = []
    hist_range = []
    selected_num = 0
    data_array = np.array(data)
    for i in range(bin_num-1):
        data_temp = data_array[np.where(data_array>=(d_min+i*stride))]
        in_num = len(np.where(data_temp<(d_min+(i+1)*stride))[0])
        hist_range.append(d_min+(i+0.5)*stride)
        hist.append(in_num)
        selected_num += in_num
    last_num = total_num - selected_num
    hist.append(last_num)
    hist_range.append(d_max - 0.5*stride)

    return hist, hist_range

matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False


data_file = open('percentage_files/head_all.txt', 'r')
data = data_file.readlines()
distance_data1 = []


for datum in data:
    distance_data1.append(((float)(datum.strip()))*100)
data_array1 = np.array(distance_data1)
data1, hist_range1 = hist_percentage(distance_data1, bin_num = 50)
"""
total_num = len(distance_data1)
data1 = np.array(distance_data1)
data_temp = data1[np.where(data1<=5.8)]
valid_num = len(np.where(data_temp>=0)[0])
print("head valid percentage: ", valid_num/total_num)
#print(len(np.where(np.array(distance_data)<=5.8)[0]))
#print(np.where(np.array(distance_data)<=5.8)[0])
"""

data_file = open('percentage_files/body_all.txt', 'r')
data = data_file.readlines()
distance_data2 = []
for datum in data:
    distance_data2.append(((float)(datum.strip()))*100)
data_array2 = np.array(distance_data2)
data2, hist_range2 = hist_percentage(distance_data2, bin_num = 50)
"""
total_num = len(distance_data2)
data2 = np.array(distance_data2)
data_temp = data2[np.where(data2<=13.6)]
valid_num = len(np.where(data_temp>5.8)[0])
print("body valid percentage: ", valid_num/total_num)
"""

# tail data part1
data_file = open('percentage_files/tail_all.txt', 'r')
data = data_file.readlines()
distance_data3 = []
for datum in data:
    distance_data3.append(((float)(datum.strip()))*100)
#print(len(distance_data))
data_array3 = np.array(distance_data3)
data3, hist_range3 = hist_percentage(distance_data3, bin_num = 50)
"""
total_num = len(distance_data3)
data3 = np.array(distance_data3)
valid_num = len(np.where(data3>26.8)[0])
print("tail valid percentage: ", valid_num/total_num)
"""

# Fit a normal distribution to the data:
mu1, std1 = norm.fit(data_array1)

mu2, std2 = norm.fit(data_array2)

mu3, std3 = norm.fit(data_array3)

print("all to show")
print(mu1, std1)
print(mu2, std2)
print(mu3, std3)

plt.figure(figsize=(6,6))
#data2 = data[np.where(data>=0)]
#print(len(np.where(data2<=5.8)[0]))

#plt.hist(data1, bins=50, normed=1, facecolor="blue", edgecolor="blue", alpha=0.7)
#plt.hist(data2, bins=50, normed=1, facecolor="red", edgecolor="red", alpha=0.7)
#plt.hist(data3, bins=50, normed=1, facecolor="green", edgecolor="green", alpha=0.7)

#plt.hist(data1, bins=30, density=True, alpha=0.6, color='b')
plt.bar(hist_range1, data1, color='r')
# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu1, std1)*100
plt.plot(x, p, 'k', linewidth=2, color='r')

#plt.hist(data2, bins=30, density=True, alpha=0.6, color='r')
plt.bar(hist_range2, data2, color='b')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu2, std2)*100
plt.plot(x, p, 'k', linewidth=2, color='b')

#plt.hist(data3, bins=30, density=True, alpha=0.6, color='g')
plt.bar(hist_range3, data3, color='g')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu3, std3)*100
plt.plot(x, p, 'k', linewidth=2, color='g')

plt.figtext(0.25, 0.6, r"$\mu_1 = 11.12\%$",
        horizontalalignment='center', fontname =font, color='r', fontsize=fontsize)
plt.figtext(0.25, 0.55, r"$s_1 = 13.31\%$",
        horizontalalignment='center', fontname =font, color='r', fontsize=fontsize)

plt.figtext(0.6, 0.7, r"$\mu_2 = 30.01\%$",
        horizontalalignment='center', fontname =font, color='b', fontsize=fontsize)
plt.figtext(0.6, 0.65, r"$s_2 = 9.59\%$",
        horizontalalignment='center', fontname =font, color='b', fontsize=fontsize)

plt.figtext(0.8, 0.48, r"$\mu_3 = 67.24\%$",
        horizontalalignment='center', fontname =font, color='g', fontsize=fontsize)
plt.figtext(0.8, 0.43, r"$s_3 = 13.93\%$",
        horizontalalignment='center', fontname =font, color='g', fontsize=fontsize)



plt.axvline(x=5, ymax=0.8, color='black', linestyle='-')
plt.figtext(0.35, 0.74, r"$p_h(5\%)$",
        horizontalalignment='center', fontname =font, color='black', fontsize=fontsize)

plt.axvline(x=30, ymax=0.9, color='black', linestyle='-')
plt.figtext(0.49, 0.82, r"$p_b(30\%)$",
        horizontalalignment='center', fontname =font, color='black', fontsize=fontsize)

plt.axvline(x=65, ymax=0.53, color='black', linestyle='-')
plt.figtext(0.675, 0.54, r"$p_t(65\%)$",
        horizontalalignment='center', fontname =font, color='black', fontsize=fontsize)

x_ticks = np.arange(-20, 120, 10)   # range from -25 to 100, show every 10
y_ticks = np.arange(0, 8, 1) # range from 0 to 20, show every 2
plt.xticks(x_ticks, fontname =font, fontsize = fontsize)
plt.yticks(y_ticks, fontname =font, fontsize = fontsize)



plt.xlabel("Touched points (MuLTAP) in percentage coordinate system $\mathbb{P}$ (%)", fontname =font,fontsize = fontsize)

plt.ylabel("Number of touched points", fontname =font, fontsize = fontsize)

#plt.title("Histogram of touching points")
plt.show()

