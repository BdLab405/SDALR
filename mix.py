import torch
import numpy as np


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_window(size, lam):
    """找到一维数据的随机裁剪区域位置（起始和结束位置）"""
    L = size[1]  # 数据的长度
    cut_idx = int(L * lam)  # 要裁剪的索引数量

    # uniform
    start_idx = np.random.randint(L - cut_idx)  # 随机裁剪起始位置
    end_idx = start_idx + cut_idx  # 裁剪结束位置

    return start_idx, end_idx  # 要裁剪区域的起始和结束位置


def cutmix_1d(batch_x, y, alpha=1.0, use_cuda=False):
    '''CutMix 数据增强技术应用于一维数据的函数
    参数:
        batch_x: [batch_size, features]，输入的一维数据批量
        y: [batch_size]，对应的标签
        alpha: float，beta分布的参数，用于生成混合权重lambda
        use_cuda: bool，是否使用CUDA
    返回:
        mixed_inputs: 混合后的输入数据
        y_a: 原始标签a
        y_b: 原始标签b
        lam: 用于混合的权重值
    '''
    assert alpha > 0
    lam = np.random.beta(alpha, alpha)  # 从 beta 分布生成 lambda 值，用于混合权重
    batch_size = batch_x.size()[0]

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # 分别获取原始标签 y 和通过随机索引选择的标签 y[index]
    y_a, y_b = y, y[index]
    # 生成随机裁剪区域的起始和结束位置
    start_idx, end_idx = rand_window(batch_x.size(), lam)
    # 将一维数据的裁剪区域替换为第二个样本对应位置的值
    batch_x[:, start_idx:end_idx] = batch_x[index, start_idx:end_idx]
    # 调整 lambda，使其与裁剪比例匹配
    lam = 1 - (end_idx - start_idx) / batch_x.size()[-1]

    return batch_x, y_a, y_b, lam


if __name__ == "__main__":

    inputs = torch.randn(32, 2048)
    targets = torch.randint(0, 9, (32,))

    mixed_x, _, _, _ = cutmix_1d(inputs, targets)
