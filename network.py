import numpy as np
import torch.nn as nn
from torchvision import models

import torch.nn.utils.weight_norm as weightNorm

import warnings
import torch

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float32(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_position, embedding_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.max_position = max_position
        self.embedding_dim = embedding_dim
        # 创建一个可训练的位置编码矩阵，形状为(max_position, embedding_dim)
        self.position_embeddings = nn.Parameter(torch.randn(max_position, embedding_dim))

    def forward(self, inputs):
        batch_size, sequence_length, _ = inputs.size()
        # 生成序列中每个位置的索引
        positions = torch.arange(0, sequence_length, dtype=torch.long, device=inputs.device)
        # 根据位置索引获取位置编码
        position_embeddings = self.position_embeddings[positions]
        # 将位置编码添加到输入序列中
        return inputs + position_embeddings.unsqueeze(0)
     

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(SinusoidalPositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)#(1,N,d)

    def forward(self, x):
        # x(B,N,d)
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class PatchEmbed1D(nn.Module):
    """
    1D 数据到 Patch 嵌入
    """

    def __init__(self, seq_length=2048, patch_size=256, in_c=1, stride_size=64, norm_layer=None):
        super(PatchEmbed1D, self).__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = (seq_length - patch_size) // stride_size + 1

        self.proj = nn.Conv1d(in_c, patch_size, kernel_size=patch_size, stride=stride_size)  # 卷积层
        self.norm = norm_layer(patch_size) if norm_layer else nn.Identity()  # 规范化层

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)  # [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = self.norm(x)
        return x


class SlidingEmbed:
    def __init__(self, input_dim, window, stride):
        self.window = window
        self.stride = stride
        self.num_patches = (input_dim - window) // stride + 1

    def __call__(self, x):
        new_x = torch.zeros((x.shape[0], self.num_patches, self.window))

        for i in range(self.num_patches):
            start_index = i * self.stride
            end_index = start_index + self.window
            new_x[:, i, :] = x[:, 0, start_index:end_index]

        return new_x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, window, stride, dropout=0.1, n_head=8, n_layers=4, p_choose="Sinusoidal", embed="SlidingEmbed"):
        super(TransformerEncoder, self).__init__()
        self.in_features = window
        self.output_dim = window
        self.patch_embed = \
            PatchEmbed1D(seq_length=input_dim, patch_size=window, in_c=1, stride_size=stride)\
            if embed == "PatchEmbed1D" else \
            SlidingEmbed(input_dim=input_dim, window=window, stride=stride)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, window))  # 初始化 class token

        self.positional_encoding = \
            SinusoidalPositionalEncoding(d_hid=window, n_position=self.num_patches + 1) \
            if p_choose == "Sinusoidal" else \
            LearnablePositionalEncoding(embedding_dim=window, max_position=self.num_patches + 1)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=window, nhead=n_head, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=n_layers)

    def forward(self, x):
        batch_size = x.size(0)  # batch_size
        x = x.unsqueeze(1) if x.dim() == 2 else x

        # 通过一维卷积将数据划分为 patch
        patches = self.patch_embed(x).cuda()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 扩展 class token
        x = torch.cat((cls_tokens, patches), dim=1)  # 在维度1上拼接 class token

        # 位置编码
        x = self.positional_encoding(x)

        seq_length = x.size(1)  # 一般来说，seq_length == num_patches + 1
        if seq_length != self.num_patches + 1:
            warnings.warn("seq_length != num_patches + 1", Warning)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        # 只取 class token
        x = x[:, 0]

        return x


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        out=x
        return out




class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        self.hidden_dim = max(bottleneck_dim//2, class_num)
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        elif type == 'MLP':
            self.fc = nn.Sequential(
                        nn.Linear(bottleneck_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, class_num),
                        nn.GELU()
        )
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class no_network:
    def train(self):
        pass
    def eval(self):
        pass