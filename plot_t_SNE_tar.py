import os
import tools
import torch
import random
import argparse

from datetime import datetime
from plot_t_SNE import prepare_dicts, plot_test_target

import numpy as np
import os.path as osp


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
    parser.add_argument('--num', type=int, default=2000)
    # 训练策略
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])

    # 神经网络（此超参数已被架空）
    parser.add_argument('--net', type=str, default='resnet18')

    # 训练轮次
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    # 计算正确率次数，你写多少，这次训练总共就会生成多少次正确率计算
    parser.add_argument("--interval", type=int, default=2)
    # 训练批次
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
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
    parser.add_argument('--username', type=str, default="WWY_PU_RES_v2")  # 一般定死  WWY_JUN_RES WWY_PU_RES_v2
    parser.add_argument('--data_name', type=str, default="PU_1d_8c_2048")  # PU_1d_8c_2048  JNU_1d_2048_2000
    parser.add_argument('--domain_names', type=list, default=['N15_M01_F10', 'N15_M07_F10', 'N15_M07_F04'])  # 一般定死
    # ['600', '800', '1000']
    # ['N15_M01_F10', 'N15_M07_F10', 'N15_M07_F04']
    parser.add_argument('--class_num', type=int, default=8)  # 一般定死
    parser.add_argument('--folder_root', type=str, default="./DATA")  # 一般定死

    args = parser.parse_args()

    args.choose_dict = ["max_epoch", "interval", "batch_size", "mix", "noise_std", "dropout", "classifier"]

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

    # 特征图片输出位置
    args.output_tar_feapic = osp.join(
        args.output,
        args.username,
        "Picture-特征图",
        datetime.strptime(args.nowtime, "%Y-%m-%d %H:%M:%S").strftime("%Y年%m月%d日%H时%M分"),
    )
    print("特征图输出位置：{}".format(args.output_tar_feapic))

    for args.s in range(len(args.domain_names)) if source_mem == -1 else [source_mem]:

        feature_dict = {}
        label_dict = {}
        score_dict = {}
        init_center_dict = {}

        for args.t in range(len(args.domain_names)) if target_mem == -1 else [args.t]:
            if args.t == args.s: continue

            print("-" * 50)
            name = "(B{})".format(args.s + 1) + args.domain_names[args.s].upper() + "→" + "(B{})".format(args.t + 1) + \
                   args.domain_names[args.t].upper()
            print("现在开始 {} 的数据收集".format(name))
            args.names = [name]

            # log&pt输出位置（所有log&pt的输出位置，里面再细分不同的领域文件）
            args.intput_tar_logpt = osp.join(
                args.output,
                args.username,
                "Target-目标域",
                "log&pt",
                name
            )
            print("模型存放位置：{}".format(args.intput_tar_logpt))

            networks = tools.read_model(args, model_path=args.intput_tar_logpt)
            print("目标模型读取成功")

            args.dset_paths = [osp.join(args.data_folder, args.domain_names[args.t] + '_label.txt')]
            print("目标域标签文件路径：{}".format(args.dset_paths[-1]))

            feature_out, label_out, score_out, init_center_out = prepare_dicts(args, networks)
            feature_dict.update(feature_out)
            label_dict.update(label_out)
            score_dict.update(score_out)
            init_center_dict.update(init_center_out)
            print("数据收集完成")
            print("-" * 50)

        print("\n开始画图\n")
        plot_test_target(feature_dict, label_dict, score_dict, init_center_dict, out_path=args.output_tar_feapic)
