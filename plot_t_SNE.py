import os
import tools
import torch
import random
import colorsys
import argparse

from datetime import datetime
from sklearn.manifold import TSNE

import numpy as np
import seaborn as sns
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_contrasting_color(color):
    r, g, b = mcolors.to_rgb(color)
    # 将颜色从 RGB 转换为 HLS
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = 1.0 - l  # 反转亮度
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return r, g, b


# 添加到字典的函数
def add_to_dict(bank, name, dictionary):
    dictionary[name] = (
        bank.cpu().numpy() if isinstance(bank, torch.Tensor) else
        bank if isinstance(bank, np.ndarray) else
        np.array(bank) if isinstance(bank, list) else
        ValueError("Unsupported type for bank")
    )


def network_data_output(args, networks):
    feature_bank, label_bank, score_bank = tools.obtain_bank(
        tools.args_data_load(args, {"test": 0.5})["test"], networks, args)

    # 添加一列全为1的向量
    feature_bank = torch.cat((feature_bank, torch.ones(feature_bank.size(0), 1)), 1)
    feature_bank = (feature_bank.t() / torch.norm(feature_bank, p=2, dim=1)).t()

    # 将数据转换为numpy数组
    feature_bank = feature_bank.float().cpu().numpy()
    label_bank = label_bank.float().cpu().numpy()
    score_bank = score_bank.float().cpu().numpy()

    # 计算初始中心位置
    init_center_bank = score_bank.T.dot(feature_bank) / (
            1e-8 + score_bank.sum(axis=0)[:, None])

    # 将得分转换为类别索引
    score_bank = np.argmax(score_bank, axis=1)

    return feature_bank, label_bank, score_bank, init_center_bank


def prepare_dicts(args, networks=None, feature_dict=None, score_dict=None, label_dict=None, init_center_dict=None,
                  name=None, feature_bank=None, score_bank=None, label_bank=None, init_center=None):
    if feature_dict is None and score_dict is None and label_dict is None and init_center_dict is None:
        feature_dict = {}
        score_dict = {}
        label_dict = {}
        init_center_dict = {}

    if networks is not None:
        # 处理每个数据集
        for name, args.dset_path in zip(args.names, args.dset_paths):
            feature_dict[name], label_dict[name], score_dict[name], init_center_dict[name] = network_data_output(args, networks)
    else:
        add_to_dict(feature_bank, name, feature_dict)
        add_to_dict(label_bank, name, label_dict)
        add_to_dict(score_bank, name, score_dict)
        add_to_dict(init_center, name, init_center_dict)

    return feature_dict, score_dict, label_dict, init_center_dict


def plot_test_target(feature_dict, label_dict, score_dict, init_center_dict, out_path=None):
    # 设置绘图参数
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    # 固定颜色方案，根据类别数生成顺序颜色列表
    classes = np.unique(next(iter(label_dict.values())))
    n_class = len(classes)  # 测试集标签类别数
    palette = sns.color_palette("tab10", n_class)  # 使用tab10配色方案

    # 步骤1：拼接两个字典中相同 key 的向量
    combined_dict = {}
    for key in feature_dict.keys():
        combined_dict[key] = np.concatenate((feature_dict[key], init_center_dict[key]), axis=0)

    # 步骤2：将拼接后的向量合并成一个大的二维张量
    dict_keys = list(combined_dict.keys())
    combined_vectors = np.concatenate(list(combined_dict.values()), axis=0)

    # 步骤3：使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, n_iter=5000)
    reduced_vectors = tsne.fit_transform(combined_vectors)
    print("t-SNE设置完成")

    # 步骤4：将降维后的结果按原字典结构拆分
    start = 0
    for key in dict_keys:
        num_samples_feature = feature_dict[key].shape[0]
        num_samples_init_center = init_center_dict[key].shape[0]

        # 恢复 feature_dict 的降维结果
        feature_dict[key] = reduced_vectors[start:start + num_samples_feature]
        start += num_samples_feature

        # 恢复 init_center_dict 的降维结果
        init_center_dict[key] = reduced_vectors[start:start + num_samples_init_center]
        start += num_samples_init_center

    # 定义标签字典
    label_dict_name = {0: "True", 1: "Predict"}

    # 遍历每个特征键
    for key_idx, key in enumerate(dict_keys):
        for dict_idx, lbl_dict in enumerate([label_dict, score_dict]):
            plt.figure(figsize=(14, 14))
            for class_idx, class_label in enumerate(classes):
                color = palette[class_idx]

                # 获取当前类别的所有图像索引
                indices = np.where(lbl_dict[key] == class_label)
                # 绘制特征散点图，设置渐变效果和灰色边框
                plt.scatter(feature_dict[key][indices, 0], feature_dict[key][indices, 1],
                            color=color, marker='o', label=class_label, s=60)
                # 绘制初始中心点
            #     plt.scatter(init_center_dict[key][class_idx, 0], init_center_dict[key][class_idx, 1], color=color,
            #                 marker=marker,
            #                 s=300, edgecolor=get_contrasting_color(color), linewidths=2, zorder=5)
            #
            # plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
            title = f"{key} ({label_dict_name[dict_idx]})"
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
            # 保存图片到指定路径
            if out_path is not None:
                if not osp.exists(out_path):
                    os.system('mkdir -p ' + out_path)
                if not osp.exists(out_path):
                    os.makedirs(out_path, exist_ok=True)
                filename = os.path.join(out_path, f"{title}.png")
                plt.savefig(filename)
            else:
                plt.show()
            plt.close()

    print("已画图")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    # 选择GPU设备
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # 选择源域
    parser.add_argument('--s', type=int, default=-1, help="source")
    # 选择目标域
    parser.add_argument('--t', type=int, default=-1, help="target")
    # 并行工作数
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    # 随机种子
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    # 当数据为0时，代替数值
    parser.add_argument('--epsilon', type=float, default=1e-5)
    # 标签平滑参数
    parser.add_argument('--smooth', type=float, default=0.1)
    # 同类型样本数量限制(-1为无限制)
    parser.add_argument('--num', type=int, default=-1)
    # 训练策略
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])

    # 神经网络（此超参数已被架空）
    parser.add_argument('--net', type=str, default='resnet18')

    # 训练轮次
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    # 计算正确率次数，你写多少，这次训练总共就会生成多少次正确率计算
    parser.add_argument("--interval", type=int, default=2)
    # 训练批次
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    # 进行Mixup的概率
    parser.add_argument('--mix', type=float, default=0)
    # 噪音干扰倍率
    parser.add_argument('--noise_std', type=float, default=0)
    # dropout概率
    parser.add_argument('--dropout', type=float, default=0.1)

    # 中介器选择（此为分类器）
    parser.add_argument('--layer', type=str, default="bn", choices=["ori", "bn", "None"])
    # 中介输出的维度(中介器为None时无视这个参数)
    parser.add_argument('--bottleneck', type=int, default=256)
    # 分类器选择（此为中介器）
    parser.add_argument('--classifier', type=str, default="wn", choices=["linear", "wn", "MLP"])

    # 输出文件夹
    parser.add_argument('--output', type=str, default='weight')

    # 数据集向导
    parser.add_argument('--username', type=str, default="WWY_PU_RES_2")  # 一般定死
    parser.add_argument('--data_name', type=str, default="PU_1d_8c_2048")
    parser.add_argument('--domain_names', type=list, default=['N15_M01_F10', 'N15_M07_F10', 'N15_M07_F04'])  # 一般定死
    # ['600', '800', '1000']
    parser.add_argument('--class_num', type=int, default=8)  # 一般定死
    parser.add_argument('--folder_root', type=str, default="./DATA")  # 一般定死

    args = parser.parse_args()

    args.choose_dict = ["max_epoch", "interval", "batch_size", "mix", "noise_std", "dropout", "classifier"]

    # args.username = 'WWY_v2'
    # args.file_name = 'PU_9c_2048'
    # args.data_name = 'PU_9c_2048'
    # names = ['N15_M01_F10', 'N15_M07_F10', 'N15_M07_F04']
    # args.class_num = 9
    #
    # folder = './DATA/PU_1d_9c_2048/'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 获取当前日期和时间
    args.nowtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.acc_list = [-1] * (len(args.domain_names) ** 2)

    args.data_folder = args.input_src_logpt = osp.join(
        args.folder_root,
        args.data_name
    )

    source_mem = args.s
    target_mem = args.t

    for args.s in range(len(args.domain_names)) if source_mem == -1 else [source_mem]:
        feature_dict = {}
        label_dict = {}
        score_dict = {}
        init_center_dict = {}

        print("\n现在开始获取源域训练集：(B{}){} ".format(args.s + 1, args.domain_names[args.s]))
        # log&pt输出位置（所有log&pt的输出位置，里面再细分不同的领域文件）
        args.output_src_logpt = osp.join(
            args.output,
            args.username,
            "Source-源域",
            "log&pt",
            "(B{})".format(args.s + 1) + args.domain_names[args.s].upper()
        )
        print("pt位置：{}".format(args.output_src_logpt))

        networks = tools.read_model(args, model_path=args.output_src_logpt)
        print("源模型读取成功")

        args.names = [("(B{})".format(args.s + 1)) + args.domain_names[args.s].upper()]
        args.dset_paths = [osp.join(args.data_folder, args.domain_names[args.s] + '_label.txt')]

        print("源域数据路径：{}".format(args.dset_paths[-1]))

        for args.t in range(len(args.domain_names)) if target_mem == -1 else [args.t]:
            if args.t == args.s: continue
            name = ("(B{})".format(args.s + 1)) + args.domain_names[args.s].upper() + "→" + ("(B{})".format(args.t + 1)) + args.domain_names[args.t].upper()
            args.dset_paths.append(osp.join(args.data_folder, args.domain_names[args.t] + '_label.txt'))
            args.names.append(name)
            print("目标域数据路径：{}".format(args.dset_paths[-1]))

        feature_out, label_out, score_out, init_center_out = prepare_dicts(args, networks)
        feature_dict.update(feature_out)
        label_dict.update(label_out)
        score_dict.update(score_out)
        init_center_dict.update(init_center_out)
        plot_test_target(feature_dict, label_dict, score_dict, init_center_dict)
