import numpy as np
import torch


class Normalize1D:
    def __call__(self, data):
        """
        对一维数据进行标准化操作

        Parameters:
        - data: 一维数据
        """
        mean = np.mean(data)
        std = np.std(data)
        normalized_data = (data - mean) / std
        return normalized_data


class Tensor1D:
    def __init__(self):
        self.type = torch.float

    def __call__(self, data):
        return torch.tensor(data, dtype=self.type)


class Bottlrneck(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class ResNet(torch.nn.Module):
    def __init__(self, in_channels=1):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),

            Bottlrneck(64, 64, 256, False),
            Bottlrneck(256, 64, 256, False),
            #
            Bottlrneck(256, 128, 512, True),
            Bottlrneck(512, 128, 512, False),

            #
            Bottlrneck(512, 256, 1024, True),
            Bottlrneck(1024, 256, 1024, False),

            #
            Bottlrneck(1024, 512, 2048, True),
            Bottlrneck(2048, 512, 2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )

        self.in_features = 2048

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 2048)
        return x

if __name__ == "__main__":
    # 使用示例：
    # 创建一个输入大小为 (batch_size, in_channels, sequence_length) 的网络实例
    model = ResNet(in_channels=1)
    print(model)