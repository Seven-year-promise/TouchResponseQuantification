import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

I = 0


def draw_sie_thre_fig():
    JI_base_path = "./ori_UNet/models-trained-on200-2/size_thre_JI_Larva"
    PC_base_path = "./ori_UNet/models-trained-on200-2/size_thre_PC_Larva"
    COLORS = ["tab:orange", "tab:olive"]
    PC = []
    JI = []
    for n in ["20", "40", "60", "80"]:
        PC_path = PC_base_path + n + ".csv"
        JI_path = JI_base_path + n + ".csv"
        PC_csv_file = pd.read_csv(PC_path, header=None)
        JI_csv_file = pd.read_csv(JI_path, header=None)
        print(JI_csv_file[1])
        # print(PC_Needle_path)
        PC += PC_csv_file[1].to_list()
        # print(len(ave_needle_accs))
        JI += JI_csv_file[1].to_list()


    thresholds = np.arange(0, len(JI))
    plt.plot(thresholds, PC, label = "PC larva for U-Net + RT + CB", color = COLORS[0])
    plt.text(12, PC[12] + 0.002, '(' + str(12) + ', ' + str(round(PC[12], 3)) +')', fontsize = 10)
    plt.scatter(12, PC[12], marker = "x", color = COLORS[0])
    plt.plot(thresholds, JI, label="JI larva for U-Net + RT + CB", color = COLORS[1])
    plt.text(12, JI[12] + 0.002, '(' + str(12) + ', ' + str(round(JI[12], 3)) + ')', fontsize=10)
    plt.scatter(12, JI[12], marker="x", color=COLORS[1])
    plt.xlabel("Threshold for size based noise filter ($T_s$)")
    plt.ylabel("JI and PC for larva segmentation")
    plt.legend(loc="upper right")
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
    draw_sie_thre_fig()
