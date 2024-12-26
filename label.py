import argparse
import torch
import numpy as np
import torch.nn as nn
from scipy.spatial.distance import cdist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter


# 创建简单的特征提取网络
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return x


# 创建简单的分类器
class Classifier(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.fc(x)


# 创建一个自定义数据集
class CustomDataset(Dataset):
    def __init__(self, num_samples, input_size, num_classes):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, input_size)  # 随机生成输入数据
        self.labels = torch.randint(0, num_classes, (num_samples,))  # 随机生成标签

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = self.data[idx]
        labels = self.labels[idx]
        return inputs, labels, idx  # 返回输入、标签和索引


# 创建 DataLoader
def create_dataloader(num_samples, input_size, num_classes, batch_size):
    dataset = CustomDataset(num_samples, input_size, num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# 定义测试用的神经网络
def create_network(input_size, hidden_size, num_classes, device):
    netF = FeatureExtractor(input_size, hidden_size).to(device)  # 将网络移到device上
    netC = Classifier(hidden_size, num_classes).to(device)       # 将分类器移到device上
    return [netF, netC]


def process_inputs(inputs, process_type):
    if process_type == "reverse":
        # 翻转数据
        inputs = torch.flip(inputs, dims=[-1])
    elif process_type == "mask":
        # 遮盖尾部1/10的样本
        length = inputs.shape[-1]
        mask_size = length // 10
        inputs[..., -mask_size:] = 0  # 将尾部1/10置为0
    elif process_type == "shift":
        # 随机位移样本前后1/5
        length = inputs.shape[-1]
        shift_size = length // 5
        shift_amount = -shift_size if np.random.choice([True, False]) else shift_size
        inputs = torch.roll(inputs, shifts=shift_amount, dims=-1)
    # 如果为 none 或者其他值则不处理
    return inputs


def check_columns(arr):
    # 获取二维数组的列数
    num_cols = arr.shape[1]
    # 初始化一个长度为列数的一维数组
    result = np.empty(num_cols, dtype=int)
    # 遍历每一列
    for i in range(num_cols):
        # 统计每列中元素的出现次数
        column = arr[:, i]
        counts = Counter(column)
        # 检查是否有数字出现次数超过3次
        found = False
        for num, count in counts.items():
            if count > 3:
                result[i] = num
                found = True
                break
        # 如果没有数字的出现次数超过3次，则设置为-1
        if not found:
            result[i] = -1
    return result


def obtain_bank(loader, networks, args, process_type):
    num_sample = len(loader.dataset)

    fea_bank = torch.randn(num_sample, args.bottleneck if args.layer != "None" else args.window)
    label_bank = torch.randn(num_sample, dtype=torch.float32)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    # 该部分不需要反向传播
    with torch.no_grad():
        for data in loader:
            inputs = data[0]
            labels = data[1]
            indx = data[-1]
            inputs = inputs.cuda()

            # 对inputs进行处理
            inputs = process_inputs(inputs, process_type)  # 根据参数进行处理

            # 将输入数据传递到 netF 进行特征提取，然后通过 netB 得到特征
            features = inputs
            for net in networks[:-1]:
                features = net(features)

            # 对特征进行归一化
            output_norm = F.normalize(features)
            # 使用 Softmax 得到分类概率
            outputs = nn.Softmax(-1)(networks[-1](features))

            # 将这些特征存储在 fea_bank 中
            fea_bank[indx] = output_norm.detach().clone().cpu()
            # 将真标签存放在 label_bank 中
            label_bank[indx] = labels.float()
            # 将分类概率存储在 score_bank 中
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    # print("仓库数据更新完成")

    return fea_bank, label_bank, score_bank


def obtain_label(loader, networks, args):
    process_type = ["none", "reverse", "mask", "shift"]
    prediction_bank = np.zeros((len(process_type), len(loader.dataset)))
    for i in range(len(process_type)):
        fea_bank, label_bank, score_bank = obtain_bank(loader, networks, args, process_type[i])
        '''
        fea_bank 包含所有测试样本的特征
        score_bank 包含所有测试样本的分类输出
        label_bank 包含所有测试样本的真实标签。'''
        # 据 softmax 输出，找到每个样本最可能的类别
        _, old_predictions = torch.max(score_bank, 1)

        # 对特征矩阵进行 L2 范数归一化，以便进行余弦相似度计算
        # 增加一个常数偏置维度可以避免某些情况下特征向量的范数为零，从而避免数值不稳定性问题。
        if args.distance == 'cosine':
            pass
            fea_bank = torch.cat((fea_bank, torch.ones(fea_bank.size(0), 1)), 1)
            fea_bank = (fea_bank.t() / torch.norm(fea_bank, p=2, dim=1)).t()

        # 将输出赋予新变量以运行后续运算
        now_predictions = old_predictions.cpu().numpy()
        # 计算分类的类别数
        K = score_bank.size(1)
        # 将特征矩阵移回 CPU 并转换为 NumPy 数组
        fea_bank = fea_bank.float().cpu().numpy()
        # 将分类输出转换为 NumPy 数组
        aff_bank = score_bank.float().cpu().numpy()

        '''迭代循环，进行了原型生成和伪标签更新的过程，这是基于原型和距离计算的类别估计'''
        for _ in range(2):
            # 计算原型的初始化位置，通过将分类概率与特征矩阵相乘
            initc = aff_bank.T.dot(fea_bank) / (1e-8 + aff_bank.sum(axis=0)[:, None])
            # 计算所有样本与初始化原型之间的距离，使用指定的距离度量方式（余弦相似度）
            dd = cdist(fea_bank, initc, args.distance)

            adjusted_dd = np.copy(dd)

            # 根据每个样本调整后的距离的最小值来更新伪标签
            pred_label = adjusted_dd.argmin(axis=1)
            now_predictions = pred_label

            # 计算最近的两个原型之间的距离差异
            min_distances = np.partition(adjusted_dd, 1, axis=1)[:, :2]
            distance_diff = np.abs(min_distances[:, 0] / min_distances[:, 1])
            # 找到距离相近的样本索引
            print(f"伪标签样本阈值倍率：{args.threshold}")
            close_samples_indices = np.where(distance_diff > args.threshold)[0]
            # 将伪标签转换为一个one-hot形式的矩阵，用于下一轮迭代
            aff_bank = np.eye(K)[now_predictions]
            aff_bank[close_samples_indices] = score_bank.float().cpu().numpy()[close_samples_indices]

        # 将选择困难症样本以-1的方式保存到predict中
        now_predictions[close_samples_indices] = -1
        prediction_bank[i] = now_predictions

    result = check_columns(prediction_bank)

    return result.astype('int')


# 测试示例
if __name__ == "__main__":
    # 参数设置
    num_samples = 100  # 样本数量
    input_size = 50  # 输入数据的维度
    hidden_size = 20  # 特征提取后的维度
    num_classes = 5  # 类别数
    batch_size = 10  # batch大小

    parser = argparse.ArgumentParser(description="LPA")
    # 特征距离选择
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    # 滑动窗口大小
    parser.add_argument('--window', type=int, default=20)
    # 伪标签样本阈值判断(趋于1时距离相近；趋于0时距离差距大，越自信)
    parser.add_argument('--threshold', type=float, default=1)

    # 中介器选择
    parser.add_argument('--layer', type=str, default="None", choices=["ori", "bn", "None"])
    # 中介输出的维度(中介器为None时无视这个参数)
    parser.add_argument('--bottleneck', type=int, default=256)
    # 分类器选择
    parser.add_argument('--classifier', type=str, default="wn", choices=["linear", "wn", "MLP"])

    args = parser.parse_args()

    args.class_num = 5

    # 检测是否有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建DataLoader
    loader = create_dataloader(num_samples, input_size, num_classes, batch_size)

    # 创建简单的网络并放入GPU
    networks = create_network(input_size, hidden_size, num_classes, device)
    _ = obtain_label(loader, networks, args)


