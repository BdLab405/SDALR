import torch
import torch.nn as nn


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes  # 类别的数量
        self.epsilon = epsilon  # 平滑参数，用于调节标签平滑的强度
        self.use_gpu = use_gpu  # 是否使用GPU加速
        self.reduction = reduction  # 是否对损失进行降维（求均值），默认为True
        self.logsoftmax = nn.LogSoftmax(dim=1)  # 创建一个 LogSoftmax 操作，用于计算对数概率

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型的原始预测矩阵（softmax之前）的形状为 (batch_size, num_classes)
            targets: 真实标签的张量，形状为 (batch_size)
        """
        log_probs = self.logsoftmax(inputs)  # 计算模型预测的对数概率(先softmax再log)
        # 将真实标签转换为one-hot形式，用于与预测概率计算损失(即：“0, 0, 0, 1, 0, 0, 0.....”形式)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:  # 如果使用 GPU，则将 targets 移至 GPU
            targets = targets.cuda()

        # 对one-hot标签进行平滑处理，即在原始标签上加入一些噪声，以提高模型的泛化性
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # 计算交叉熵损失
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:  # 如果 reduction 为 True，则返回损失的均值
            return loss.mean()
        else:  # 否则，返回未降维的损失
            return loss

