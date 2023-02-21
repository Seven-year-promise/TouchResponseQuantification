import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

I = 0


def draw_recall_correct_fig():
    base_path = "./Eval-All-Methods/"

    recall_same_path = "recall_ratio.csv"
    correct_same_path = "correct_ratio.csv"
    methods = ["binarization/", "otsu/", "lrb/", "rg/", "u-net/"]
    # markers = [".", "s", "*", "h"]
    labels = ["Thre", "Otsu", "LRb", "RGb", "U-Net"]

    COLORS = ["tab:green", "tab:red", "tab:purple", "tab:brown", "tab:orange"]
    recalls_methods = []
    corrects_methods = []
    thresholds = []
    # recall_lines = []
    # correct_lines = []
    for m, l, c in zip(methods, labels, COLORS):
        recalls_thre = []
        corrects_thre = []
        recall_path = base_path + m + recall_same_path
        correct_path = base_path + m + correct_same_path
        recall_csv_file = pd.read_csv(recall_path, header=None).values
        correct_csv_file = pd.read_csv(correct_path, header=None).values

        thresholds = correct_csv_file[0, 1:]
        # print(thresholds)
        for i, t in enumerate(thresholds):
            recall_ratios_list = np.array(recall_csv_file[1:, i + 1], dtype=np.float)
            correct_ratios_list = np.array(correct_csv_file[1:, i + 1], dtype=np.float)
            # print(correct_ratios_list)
            recall_ratio_ave = np.average(recall_ratios_list[np.where(recall_ratios_list >= 0)])
            correct_ratio_ave = np.average(correct_ratios_list[np.where(correct_ratios_list >= 0)])
            recalls_thre.append(recall_ratio_ave)
            corrects_thre.append(correct_ratio_ave)
        recalls_methods.append(recalls_thre)
        corrects_methods.append(corrects_thre)

    for r, l, c in zip(recalls_methods, labels, COLORS):
        plt.plot(thresholds, r, label=l, color=c)

    plt.xlabel("Threshold of IOU ($T_{IOU}$)", fontname = "Arial", fontsize=18)
    plt.ylabel("Ratio of recall ($R_r$)", fontname = "Arial", fontsize=18)

    plt.xticks(fontsize=18, fontname="Arial")
    plt.yticks(fontsize=18, fontname="Arial")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("./plots/comparison_rr.eps", format='eps')

    for correct, l, c in zip(corrects_methods, labels, COLORS):
        plt.plot(thresholds, correct, label=l, color=c)
    plt.xlabel("Threshold of IOU ($T_{IOU}$)", fontname = "Arial", fontsize=18)
    plt.ylabel("Ratio of precision ($R_p$)", fontname = "Arial", fontsize=18)

    plt.xticks(fontsize=18, fontname="Arial")
    plt.yticks(fontsize=18, fontname="Arial")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("./plots/comparison_rp.eps", format='eps')


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


if __name__ == '__main__':
    draw_recall_correct_fig()
