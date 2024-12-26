import os
import numpy as np
import matplotlib.pyplot as plt
import random


def find_txt_files(folder_path):
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files


def parse_filename(filename):
    parts = filename.split('_')
    condition = '_'.join(parts[:-1])
    fault_code = parts[-1]
    return condition, fault_code


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    data = [float(item) for line in data for item in line.strip().split()]
    return data


def plot_vibration(data_samples, condition, output_folder):
    plt.figure(figsize=(10, 8))
    for i, data in enumerate(data_samples):
        plt.subplot(5, 4, i + 1)  # 5 rows, 4 columns for 20 samples
        plt.plot(data)
        plt.title(f'Sample {i + 1}')
    plt.suptitle(f'{condition}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_folder, f'{condition}.png')
    plt.savefig(output_path)
    plt.close()


def main(input_folder, output_folder):
    txt_files = find_txt_files(input_folder)
    file_groups = {}

    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        condition, fault_code = parse_filename(filename)
        key = condition
        if key not in file_groups:
            file_groups[key] = []
        file_groups[key].append(txt_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for condition, files in file_groups.items():
        sample_files = random.sample(files, min(20, len(files)))
        data_samples = [read_txt_file(file) for file in sample_files]
        numpy_samples = np.array(data_samples)
        # 计算每一列的均值,计算每一列的标准差,进行标准化
        # means = np.mean(numpy_samples, axis=0)
        # std_devs = np.std(numpy_samples, axis=0)
        # standardized_data = (numpy_samples - means) / std_devs

        # 计算每一列的最小值,计算每一列的最大值,进行Min-Max归一化并使用列表表达式
        # mins = np.min(numpy_samples, axis=0)
        # maxs = np.max(numpy_samples, axis=0)
        # standardized_data = 2 * ((numpy_samples - mins) / (maxs - mins)) - 1

        # 不做任何变化
        standardized_data = numpy_samples

        standardized_data = standardized_data.tolist()

        plot_vibration(standardized_data, condition, output_folder)


# 使用方法
input_folder = r'D:\项目\AaD+伪\DATA\JNU_2s_2048_2000\data'
output_folder = r'./pic'
main(input_folder, output_folder)
