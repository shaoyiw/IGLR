import os
import sys
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from isegm.utils.exp import load_config_file


# the folder must follow the structure like "data_path/model_name/dataset_name.pickle"
def main():

    data_path = '/home/ubuntu/liu/Experiment_data/plot_iou_20'  # this folder contains all the data
    dataset_names = ['SBD', 'DAVIS', 'Berkeley', 'GrabCut']
    result_save_path = '/home/ubuntu/liu/Experiment_data/per_click_iou'
    # path save all model's dataset's data

    files_list = collect_dataset_files(data_path, dataset_names)
    # 对每个数据集进行绘制
    for dataset_name in dataset_names:
        plt.figure(figsize=(12, 7))
        max_click = 0
        plot_datas_paths = files_list[dataset_name]  # n models with n datas
        for plot_datas_path in plot_datas_paths:
            with open(plot_datas_path, 'rb') as f:
                data = pickle.load(f)
            model_name = plot_datas_path.split('/')[-2]
            if model_name == 'FocusCut':
                per_click_iou = np.array(data['mean_ious'])  # (n_sample,n_click)
            elif model_name in ['f-BRS-B', 'BRS']:
                per_click_iou = np.array(data).mean(0)
            else:
                per_click_iou = np.array(data['all_ious']).mean(0)  # (n_sample,n_click)
            max_click = len(per_click_iou)
            plt.plot(1 + np.arange(max_click), per_click_iou, linewidth=2, label=model_name)

        plt.title(f'{dataset_name}', fontsize='x-large')
        plt.grid()
        plt.legend(loc=4, fontsize='x-large')
        plt.yticks(fontsize='x-large')
        plt.xticks(1 + np.arange(max_click), fontsize='x-large')
        plt.savefig(result_save_path + '/' + dataset_name + '.png')
        plt.close()
def collect_dataset_files(root_dir, dataset_names):
    """
    收集多个数据集的所有模型对应的.pickle文件路径，按数据集分类

    :param root_dir: 根目录（例如 'plot_lou_20'）
    :param dataset_names: 数据集名称列表（例如 ['DAVIS', 'Berkeley']）
    :return: 字典格式结果，键为数据集名称，值为对应文件路径列表
    """
    file_dict = {dataset: [] for dataset in dataset_names}
    # 遍历根目录下的所有子目录（模型名称）
    for model_dir in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        for filename in os.listdir(model_path):
            file_path = os.path.join(model_path, filename)
            # 仅处理.pickle文件
            if os.path.isfile(file_path) and (filename.endswith(".pickle") or filename.endswith(".pkl")):
                # 检查文件名是否以任一数据集名称开头
                for dataset in dataset_names:
                    if filename.startswith(dataset):
                        file_dict[dataset].append(file_path)
                        break  # 匹配到第一个数据集后退出循环
    return file_dict


def get_target_file_path(plots_path, dataset_name):
    previous_plots = sorted(plots_path.glob(f'{dataset_name}_*.png'))
    if len(previous_plots) == 0:
        index = 0
    else:
        index = int(previous_plots[-1].stem.split('_')[-1]) + 1

    return str(plots_path / f'{dataset_name}_{index:03d}.png')


if __name__ == '__main__':
    main()
