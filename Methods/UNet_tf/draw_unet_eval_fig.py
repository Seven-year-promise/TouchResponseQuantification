import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
I = 0
font_size=18
font_name="Arial"

def draw_eval_fig():
    model_dir = "./ori_UNet/performance/"
    eval_csv_results= ["random_contrast",
                   "random_contrast_gaussian_noise",
                   "gaussian_noise",
                   "without_augmentation",
                   "random_rotation_and_contrast",
                   "random_rotation_contrast_gaussian_noise",
                   "random_rotation",
                   "random_rotation_gaussian_noise"]
    model_types = ["U-Net + CB",
                   "U-Net + CB + GN",
                   "U-Net + GN",
                   "U-Net",
                   "U-Net + RT + CB",
                   "U-Net + RT + CB + GN",
                   "U-Net + RT",
                   "U-Net + RT + GN"]

    eval_csv_results = ["without_augmentation",
                        "random_rotation",
                        "random_contrast",
                        "gaussian_noise",
                        "random_rotation_and_contrast",
                        "random_contrast_gaussian_noise",
                        "random_rotation_gaussian_noise",
                        "random_rotation_contrast_gaussian_noise"]
    model_types = ["U-Net",
                   "U-Net + RT",
                   "U-Net + CB",
                   "U-Net + GN",
                   "U-Net + RT + CB",
                   "U-Net + CB + GN",
                   "U-Net + RT + GN",
                   "U-Net + RT + CB + GN"]

    COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:olive"]
    """
    fig00, axs00 = plt.subplots(1, 1)
    fig01, axs01 = plt.subplots(1, 1)
    fig10, axs10 = plt.subplots(1, 1)
    fig11, axs11 = plt.subplots(1, 1)
    lines_axis00 = []
    lines_axis01 = []
    lines_axis10 = []
    lines_axis11 = []
    """
    lines = []
    for model_type, color, eval_file_name in zip(model_types, COLORS, eval_csv_results):
        PC_Needle_path = model_dir + eval_file_name + "_PC_Needle.csv"
        JI_Needle_path = model_dir + eval_file_name + "_JI_Needle.csv"
        PC_Larva_path = model_dir + eval_file_name + "_PC_Larva.csv"
        JI_Larva_path = model_dir + eval_file_name + "_JI_Larva.csv"
        PC_Needle_csv_file = pd.read_csv(PC_Needle_path, header=None)
        JI_Needle_csv_file = pd.read_csv(JI_Needle_path, header=None)
        PC_Larva_csv_file = pd.read_csv(PC_Larva_path, header=None)
        JI_Larva_csv_file = pd.read_csv(JI_Larva_path, header=None)
        #print(PC_Needle_path)
        ave_needle_accs = PC_Needle_csv_file[1].to_list()
        #print(len(ave_needle_accs))
        ave_fish_accs = JI_Needle_csv_file[1].to_list()
        ave_needle_ius = PC_Larva_csv_file[1].to_list()
        ave_fish_ius = JI_Larva_csv_file[1].to_list()

        epoches = np.arange(1, len(ave_needle_accs) +1) * 500
        #line00, = axs00.plot(epoches, ave_needle_accs, color=color)
        #lines_axis00.append(line00)
        if I == 0:
            line, = plt.plot(epoches, ave_needle_accs, label=model_type, color=color)
        epoches = np.arange(1, len(ave_fish_accs) +1) * 500
        #line01, = axs01.plot(epoches, ave_needle_ius, color=color)
        #lines_axis01.append(line01)
        if I == 1:
            line, = plt.plot(epoches, ave_needle_ius, label=model_type, color=color)
        epoches = np.arange(1, len(ave_needle_ius) +1) * 500
        #line10, = axs10.plot(epoches, ave_fish_accs, color=color)
        #lines_axis10.append(line10)
        if I == 2:
            line, = plt.plot(epoches, ave_fish_accs, label=model_type, color=color)
        epoches = np.arange(1, len(ave_fish_ius) +1) * 500
        #line11, = axs11.plot(epoches, ave_fish_ius, color=color)
        #lines_axis11.append(line11)
        if I == 3:
            line, = plt.plot(epoches, ave_fish_ius, label=model_type, color=color)
        lines.append(line)
    """
    axs00.set_ylabel('PC Needle')
    axs00.set_xlabel("Training Epoch")
    axs[0, 1].set_ylabel('JI Needle')
    axs[0, 1].set_xlabel("Training Epoch")
    axs[1, 0].set_ylabel('PC Larva')
    axs[1, 0].set_xlabel("Training Epoch")
    axs[1, 1].set_ylabel('JI Larva')
    axs[1, 1].set_xlabel("Training Epoch")
    """
    plt.xlabel("Training Epoch", fontsize=font_size, fontname=font_name)
    plt.xticks(fontsize=font_size, fontname=font_name)
    plt.yticks(fontsize=font_size, fontname=font_name)
    if I == 0:
        plt.ylabel("PC (Needle)", fontsize=font_size, fontname=font_name)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("./plots/pc_needle.png", format='png')
    elif I == 1:
        plt.ylabel("JI (Needle)", fontsize=font_size, fontname=font_name)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("./plots/ji_needle.png", format='png')
    elif I == 2:
        plt.ylabel("PC (Larva)", fontsize=font_size, fontname=font_name)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("./plots/pc_larva.png", format='png')
    else:
        plt.ylabel("JI (Larva)", fontsize=font_size, fontname=font_name)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("./plots/ji_larva.png", format='png')


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

if __name__ == '__main__':
    draw_eval_fig()
