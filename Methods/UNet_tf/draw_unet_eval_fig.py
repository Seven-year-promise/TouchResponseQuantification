import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def draw_eval_fig():
    model_dir = "./OriUNet/models-trained-on200/"
    eval_csv_results= ["random_contrast",
                   "random_contrast_gaussian_noise",
                   "without_augmentation",
                   "random_rotation_and_contrast",
                   "random_rotation_contrast_gaussian_noise",
                   "random_rotation",
                   "random_rotation_gaussian_noise"]
    model_types = ["random contrast",
                   "random contrast and gaussian noise",
                   "without augmentation",
                   "random rotation and contrast",
                   "random rotation and contrast and gaussian noise",
                   "random rotation",
                   "random rotation and gaussian noise"]

    COLORS = ["b", "g", "r", "c", "m", "y", "k"]
    fig, axs = plt.subplots(2, 2)
    lines_axis00 = []
    lines_axis01 = []
    lines_axis10 = []
    lines_axis11 = []

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
        line00, = axs[0, 0].plot(epoches, ave_needle_accs, color=color)
        lines_axis00.append(line00)
        epoches = np.arange(1, len(ave_fish_accs) +1) * 500
        line01, = axs[0, 1].plot(epoches, ave_needle_ius, color=color)
        lines_axis01.append(line01)
        epoches = np.arange(1, len(ave_needle_ius) +1) * 500
        line10, = axs[1, 0].plot(epoches, ave_fish_accs, color=color)
        lines_axis10.append(line10)
        epoches = np.arange(1, len(ave_fish_ius) +1) * 500
        line11, = axs[1, 1].plot(epoches, ave_fish_ius, color=color)
        lines_axis11.append(line11)

    axs[0, 0].set_ylabel('PC Needle')
    axs[0, 0].set_xlabel("Training Epoch")
    axs[0, 1].set_ylabel('JI Needle')
    axs[0, 1].set_xlabel("Training Epoch")
    axs[1, 0].set_ylabel('PC Larva')
    axs[1, 0].set_xlabel("Training Epoch")
    axs[1, 1].set_ylabel('JI Larva')
    axs[1, 1].set_xlabel("Training Epoch")

    fig.legend(lines_axis00, model_types, 'upper right')
    fig.legend(lines_axis01, model_types, 'upper right')
    fig.legend(lines_axis10, model_types, 'upper right')
    fig.legend(lines_axis11, model_types, 'upper right')

    #plt.tight_layout()
    plt.show()
    plt.legend(labels=["PC Needle", "JI Needle", "", "JI Larva"], loc="best")

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
