import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

I = 0


def draw_recall_correct_fig():
    recall_base_path = "./ori_UNet/models-trained-on200-2/larva_recall_ratio"
    correct_base_path = "./ori_UNet/models-trained-on200-2/larva_correct_ratio"

    markers = [".", "s", "*", "h"]
    labels = ["2 larvae", "3 larvae", "4 larvae", "5 larvae"]

    COLORS = ["tab:green", "tab:red", "tab:purple", "tab:brown"]
    recalls_num = []
    corrects_num = []
    thresholds = []
    #recall_lines = []
    #correct_lines = []
    for n, l, m, c in zip(range(2, 6), labels, markers, COLORS):
        recalls_thre = []
        corrects_thre = []
        recall_path = recall_base_path + str(n) + ".csv"
        correct_path = correct_base_path + str(n) + ".csv"
        recall_csv_file = pd.read_csv(recall_path, header=None).values
        correct_csv_file = pd.read_csv(correct_path, header=None).values

        thresholds = correct_csv_file[0, 1:]
        #print(thresholds)
        for i, t in enumerate(thresholds):
            recall_ratios_list = np.array(recall_csv_file[1:, i+1], dtype=np.float)
            correct_ratios_list = np.array(correct_csv_file[1:, i + 1], dtype=np.float)
            #print(correct_ratios_list)
            recall_ratio_ave = np.average(recall_ratios_list[np.where(recall_ratios_list >= 0)])
            correct_ratio_ave = np.average(correct_ratios_list[np.where(correct_ratios_list >= 0)])
            recalls_thre.append(recall_ratio_ave)
            corrects_thre.append(correct_ratio_ave)
        recalls_num.append(recalls_thre)
        corrects_num.append(corrects_thre)

    for r, l, m, c in zip(recalls_num, labels, markers, COLORS):
        plt.plot(thresholds, r, label=l, color=c)

    plt.xlabel("Threshold of IOU ($T_{IOU}$)")
    plt.ylabel("Ratio of recall ($R_r$)")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    for correct, l, m, c in zip(corrects_num, labels, markers, COLORS):
        plt.plot(thresholds, correct, label=l, color=c)
    plt.xlabel("Threshold of IOU ($T_{IOU}$)")
    plt.ylabel("Ratio of precision ($R_p$)")

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

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
