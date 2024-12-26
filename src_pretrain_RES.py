import argparse
import time
import os
import json
import random
import tools
import torch
import plot_t_SNE

import numpy as np
import os.path as osp
import torch.optim as optim

from datetime import datetime
from loss import CrossEntropyLabelSmooth
from tools import Entropy, op_copy, lr_scheduler, print_args, print_excel, write_to_excel, find_position
from data_list import txt_loader, make_dataset


# 源域训练函数
def train_source(args, dset_loaders):
    ## 网络初始化
    networks = tools.read_model(args)

    ## 优化器设置
    param_group = []  # 创建一个空列表，用于存储网络模型中的参数组信息
    learning_rate = args.lr  # 从超参数中获取学习率
    '''
    遍历网络的所有参数
    将每个参数及其对应的学习率添加到 param_group 列表中'''
    for idx, net in enumerate(networks):
        for k, v in net.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate * args.lr_f[idx]}]

    optimizer = optim.SGD(param_group)  # 使用 optim.SGD 类创建一个随机梯度下降（SGD）优化器，将参数组 param_group 传递给优化器
    optimizer = op_copy(optimizer)

    ## 参数初始化
    acc_init = 0
    iter_num = 0
    print_loss = 0
    mix_train = tools.mix_train()
    max_iter = args.max_epoch * len(dset_loaders["train"])
    interval_iter = max_iter // args.interval
    over_time = time.time()

    for net in networks:
        net.train()

    ## 迭代训练过程
    for epoch in range(args.max_epoch):
        iter_source = iter(dset_loaders["train"])
        for inputs_source, labels_source, _ in iter_source:

            '''
            检查当前批次的样本数量是否为 1，如果是则跳过这个批次。
            这可能是为了避免在批次大小为 1 时出现问题，因为某些操作（如 Batch Normalization）可能需要更多的样本。'''
            if inputs_source.size(0) == 1:
                continue

            # 更新迭代次数，用于学习率调度
            iter_num += 1
            # 学习率更新
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

            mix_classifier_losss = mix_train(args, networks, inputs_source, labels_source)

            feature_src = inputs_source
            for net in networks[:-1]:
                feature_src = net(feature_src)

            # 分类器 用于对处理后的特征进行分类，得到源域的分类结果
            outputs_source = networks[-1](feature_src)

            classifier_losss = torch.nn.CrossEntropyLoss(reduction='none')(outputs_source, labels_source)

            if torch.mean(mix_classifier_losss) == 0:
                final_classifier_loss = torch.mean(classifier_losss)
            elif torch.mean(classifier_losss) == 0:
                final_classifier_loss = torch.mean(mix_classifier_losss)
            else:
                final_classifier_loss = torch.mean(torch.cat((mix_classifier_losss, classifier_losss), dim=0))

            print_loss = final_classifier_loss
            optimizer.zero_grad()
            final_classifier_loss.backward()
            optimizer.step()

        # 每轮次之后计算loss
        print("轮次:{:·>3d}/{:·>3d} || loss:{:4.3f} || 耗时：{:4.2f}s".format(epoch + 1,
                                                                             args.max_epoch,
                                                                             print_loss,
                                                                             time.time() - over_time))

        ## 性能评估
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for net in networks:
                net.eval()

            # 开始计算正确率
            acc_s_te, acc_matrix, _ = tools.cal_acc(dset_loaders['val'], networks)
            log_str = '任务: {}; mix批次:{:·>5d}; 批次:{:·>5d}/{:·>5d}; 总正确率 = {:4.2f}%'. \
                          format(args.name_src, mix_train.mix_iter_num, iter_num, max_iter, acc_s_te) + '\n'
            correct_rate, acc_str = tools.print_acc(acc_matrix)
            log_str += acc_str

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                ## 更新正确率
                acc_init = acc_s_te

                ## 保存模型
                for idx, net in enumerate(networks[:-1], start=1):
                    best_net = net.state_dict()
                    torch.save(best_net, osp.join(args.output_src_logpt, f"source_{idx}.pt"))

                best_netC = networks[-1].state_dict()
                torch.save(best_netC, osp.join(args.output_src_logpt, "source_C.pt"))

            for net in networks:
                net.train()

        over_time = time.time()

        ## 测试自家测试集
        if iter_num == max_iter:
            for idx, net in enumerate(networks[:-1], start=1):
                net.load_state_dict(torch.load(f"{args.output_src_logpt}/source_{idx}.pt"))
            networks[-1].load_state_dict(torch.load(f"{args.output_src_logpt}/source_C.pt"))

            for net in networks:
                net.eval()

            acc_s_te, acc_matrix, _ = tools.cal_acc(dset_loaders['test'], networks)
            log_str = '测试集任务: {}; 总正确率 = {:4.2f}%'.format(args.name_src, acc_s_te) + '\n'
            correct_rate, acc_str = tools.print_acc(acc_matrix)
            log_str += acc_str

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            # 将测试集成绩记录在案
            args.class_acc_list = [np.mean(correct_rate)] + correct_rate
            args.acc_list[args.s] = acc_s_te

            for net in networks:
                net.train()

    return networks


# 测试目标与函数
def btest_target(args, dset_loaders):
    ## 加载网络
    networks = tools.read_model(args, args.output_src_logpt)

    acc, acc_matrix, _ = tools.cal_acc(dset_loaders['test'], networks)
    log_str = '任务: {}, 正确率 = {:.2f}%'.format(args.name, acc) + '\n'
    correct_rate, acc_str = tools.print_acc(acc_matrix)
    log_str += acc_str

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    args.acc_list[args.find_k(args.s, args.t) + 3] = acc
    print("测试目标数据集结果如下：")
    print(log_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    # 选择GPU设备
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # 选择源域
    parser.add_argument('--s', type=int, default=-1, help="source")
    # 选择目标域
    parser.add_argument('--t', type=int, default=-1, help="target")
    # 并行工作数
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    # 随机种子
    parser.add_argument('--seed', type=int, default=2024, help="random seed")
    # 当数据为0时，代替数值
    parser.add_argument('--epsilon', type=float, default=1e-5)
    # 标签平滑参数
    parser.add_argument('--smooth', type=float, default=0.1)
    # 同类型样本数量限制(-1为无限制)
    parser.add_argument('--num', type=int, default=-1)

    # 神经网络（此超参数已被架空）
    parser.add_argument('--net', type=str, default='resnet18')

    # 训练轮次
    parser.add_argument('--max_epoch', type=int, default=1, help="max iterations")
    # 计算正确率次数，你写多少，这次训练总共就会生成多少次正确率计算
    parser.add_argument("--interval", type=int, default=1)
    # 训练批次
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    # 学习率
    parser.add_argument('--lr', type=float, default=7e-3, help="learning rate")
    # 学习率倍率（仅影响特征提取）
    parser.add_argument('--lr_f', type=str, default="[1, 1, 1]")
    # 进行Mixup的概率
    parser.add_argument('--mix', type=float, default=0)
    # 噪音干扰倍率
    parser.add_argument('--noise_std', type=float, default=0.1)
    # dropout概率
    parser.add_argument('--dropout', type=float, default=0.1)

    # 中介器选择
    parser.add_argument('--layer', type=str, default="bn", choices=["ori", "bn", "None"])
    # 中介输出的维度(中介器为None时无视这个参数)
    parser.add_argument('--bottleneck', type=int, default=256)
    # 分类器选择
    parser.add_argument('--classifier', type=str, default="wn", choices=["linear", "wn", "MLP"])

    # 输出文件夹
    parser.add_argument('--output', type=str, default='weight')

    # 数据集向导
    parser.add_argument('--username', type=str, default="WWY_JUN")  # 一般定死 "WWY_PU" "WWY_JUN"
    parser.add_argument('--data_name', type=str, default="JNU_1d_2048_2000")  # "PU_1d_8c_2048" "JNU_1d_2048_2000"
    parser.add_argument('--domain_names', type=list, default=['600', '800', '1000'])  # 一般定死
    # ['N15_M01_F10', 'N15_M07_F10', 'N15_M07_F04']
    # ['600', '800', '1000']
    parser.add_argument('--class_num', type=int, default=4)  # 一般定死 8 4
    parser.add_argument('--folder_root', type=str, default="./DATA")  # 一般定死

    args = parser.parse_args()

    args.choose_dict = ["max_epoch", "interval", "batch_size", "lr",
                        "lr_f", "mix", "noise_std", "dropout", "classifier"]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # 获取当前日期和时间
    args.nowtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    args.acc_list = [-1] * (len(args.domain_names) ** 2)

    # 修改学习率倍率格式
    args.lr_f = json.loads(args.lr_f)

    # 获得文件完整路径
    args.data_folder = args.input_src_logpt = osp.join(
        args.folder_root,
        args.data_name
    )

    source_mem = args.s
    target_mem = args.t
    for args.s in range(len(args.domain_names)) if source_mem == -1 else [source_mem]:

        print("\n\n现在开始拿源域训练集：(B{}){} 进行训练".format(args.s + 1, args.domain_names[args.s]))
        args.dset_path = osp.join(args.data_folder, args.domain_names[args.s] + '_label.txt')
        print("源域读取路径：{}".format(args.dset_path))

        # excel输出位置（所有excel，包括总结excel和不同领域的细分excel）
        args.output_src_excel = osp.join(
            args.output,
            args.username,
            "Source",
        )
        print("excel输出位置：{}".format(args.output_src_excel))

        # log&pt输出位置（所有log&pt的输出位置，里面再细分不同的领域文件）
        args.output_src_logpt = osp.join(
            args.output_src_excel,
            "log&pt",
            "(B{})".format(args.s + 1) + args.domain_names[args.s].upper()
        )
        print("log&pt输出位置：{}".format(args.output_src_logpt))

        # 特征图片输出位置
        args.output_src_feapic = osp.join(
            args.output,
            args.username,
            "Picture",
            datetime.strptime(args.nowtime, "%Y-%m-%d %H:%M:%S").strftime("%Y年%m月%d日%H时%M分"),
            "(B{})".format(args.s + 1) + args.domain_names[args.s].upper()
        )
        print("特征图片输出位置：{}".format(args.output_src_feapic))

        args.name_src = ("(B{})".format(args.s + 1)) + args.domain_names[args.s].upper()
        args.names = [("Before" + args.name_src)]
        args.dset_paths = [args.dset_path]
        # 创建文件夹
        # 直接创建log&pt的对应领域的文件夹，就会包含所有需要的路径了
        if not osp.exists(args.output_src_logpt):
            os.system('mkdir -p ' + args.output_src_logpt)
        if not osp.exists(args.output_src_logpt):
            os.makedirs(args.output_src_logpt, exist_ok=True)

        args.out_file = open(osp.join(args.output_src_logpt, 'log_{}.txt'.format(
            args.nowtime.replace("-", "").replace(":", "").replace(" ", "_"))), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()

        print("源域训练开始")
        args.class_acc_list = [-1] * (args.class_num + 1)
        args.find_k = find_position(len(args.domain_names))

        data_loaders = tools.args_data_load(args, {"train": 0.7, "val": 0.1, "test": 0.2})
        args.input_dim = len(txt_loader(make_dataset(open(args.dset_path).readlines()[:1], None)[0][0]))
        train_source(args, data_loaders)

        str_list = ["B{}".format(args.s + 1)]
        str_list.extend(['类型{}'.format(y + 1) for y in range(args.class_num)])
        write_to_excel(print_excel(args, str_list, args.class_acc_list),
                       args.output_src_excel + f'/(B{args.s + 1})src_class_output.xlsx')
        print("源域训练结束")

        print("\n训练好之后，测试目标域")
        for args.t in range(len(args.domain_names)) if target_mem == -1 else [args.t]:
            if args.t == args.s: continue

            args.name = ("(B{})".format(args.s + 1)) + args.domain_names[args.s].upper() + "→" + ("(B{})".format(args.t + 1)) + \
                        args.domain_names[args.t].upper()
            args.names.append("Before" + args.name)
            args.dset_path = osp.join(args.data_folder, args.domain_names[args.t] + '_label.txt')
            args.dset_paths.append(args.dset_path)
            print("目标域路径：{}".format(args.dset_path))

            data_loaders = tools.args_data_load(args, {"test": 0.2})
            btest_target(args, data_loaders)

        # 生成参考特征图片
        # feature_dict, label_dict, score_dict, init_center_dict = plot_t_SNE.prepare_dicts(args, tools.read_model(args, args.output_src_logpt))
        # plot_t_SNE.plot_test_target(feature_dict, label_dict, score_dict, init_center_dict, args.output_src_feapic)


    # 生成列表
    str_list = ["B{}".format(x + 1) for x in range(len(args.domain_names))]
    str_list.extend(['B{}→B{}'.format(x + 1, y + 1) for x in range(len(args.domain_names)) for y in range(len(args.domain_names)) if x != y])
    # 写入数据到 Excel 文件
    write_to_excel(print_excel(args, str_list, args.acc_list), args.output_src_excel + '/src_output.xlsx')
