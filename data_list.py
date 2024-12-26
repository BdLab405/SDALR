#from __future__ import print_function, division

import numpy as np

from PIL import Image
from torch.utils.data import Dataset


# 该函数接受图像路径列表 image_list 和标签列表 labels，生成一个包含图像路径及其对应标签的列表 images。
def make_dataset(data_list, labels):
    if labels:  # 如果提供了标签 (labels)，则构建元组列表，每个元组包含图像路径和对应标签
      len_ = len(data_list)
      datas = [(data_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:  # 如果没有提供标签，函数将解析图像列表，假定格式为路径 + 标签，生成元组列表
      if len(data_list[0].split()) > 2:
        datas = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in data_list]
      else:
        datas = [(val.split()[0], int(val.split()[1])) for val in data_list]
    return datas


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def txt_loader(path):
    try:
        # 打开文件并读取内容
        with open(path, 'r') as file:
            content = file.read()
        # 将文本内容分割成字符串列表
        values_as_strings = content.split()
        # 将字符串列表转换为数字列表
        numeric_values = [float(value) for value in values_as_strings]
        # 将数字列表转换为一维数组
        one_dimensional_array = np.array(numeric_values)
        return one_dimensional_array
    except Exception as e:
        print(f"Error: {e}")
        return None


class DataList(Dataset):
    def __init__(self, data_list, labels=None, transform=None, target_transform=None, mode='TXT'):
        contents = make_dataset(data_list, labels)  # contents是一个包含所有[数据, 标签]的元组列表
        if len(contents) == 0:
            raise(RuntimeError("Found 0 data in subfolders" + "\n"
                               "Supported data extensions "))

        # 根据模式选择加载器函数
        loader_dict = {'RGB': rgb_loader, 'L': l_loader, 'TXT': txt_loader}

        self.loader = loader_dict.get(mode)
        self.contents = contents
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # path 为数据路径
        # target 为标签
        path, target = self.contents[index]
        data = self.loader(path)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target, index

    def __len__(self):
        return len(self.contents)


class DataList_idx(Dataset):
    def __init__(self, data_list, labels=None, transform=None, target_transform=None, mode='TXT'):
        contents = make_dataset(data_list, labels)
        if len(contents) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + "\n"
                               "Supported image extensions are: "))

        self.contents = contents

        # 根据模式选择加载器函数
        loader_dict = {'RGB': rgb_loader, 'L': l_loader, 'TXT': txt_loader}
        # self.loader = loader_dict.get(mode)
        #
        # if transform is not None:
        #     paths, _ = zip(*contents)
        #     datas = []
        #     for path in paths:
        #         data = self.loader(path)
        #         datas.extend(data)
        #     datas = np.array(datas)
        #     self.transform = transform(datas, std=std, ctype=mode, choose=choose)
        # else:
        #     self.transform = transform
        #
        # self.target_transform = target_transform

        self.loader = loader_dict.get(mode)
        self.contents = contents
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.contents[index]
        data = self.loader(path)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target, index

    def __len__(self):
        return len(self.contents)


class ClearDataList_idx(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]

        return data, label, index

    def __len__(self):
        return self.datas.shape[0]
