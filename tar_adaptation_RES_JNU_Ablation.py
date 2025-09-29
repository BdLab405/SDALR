import argparse
import json
import time
import os
import text
import plot_t_SNE
import CenterBasedClustering
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from datetime import datetime

import loss
import random
import torch.nn.functional as F
import tools
from tools import op_copy, lr_scheduler, print_args, print_excel, write_to_excel, find_position
from data_list import txt_loader, make_dataset


def obtain_label(loader, networks, args):
    if args.mix == 1:
        process_type = ["none"]  # "reverse", "mask", "shift",
    elif args.mix == 2:
        process_type = ["reverse", "mask", "shift", "none"]  # "reverse", "mask", "shift",
    else:
        a = b
    prediction_bank = np.zeros((len(process_type), len(loader.dataset)))
    for i in range(len(process_type)):
        print(f"样本变换：{process_type[i]}")
        fea_bank, label_bank, score_bank = tools.obtain_bank(loader, networks, args, process_type[i])
        '''
        fea_bank 包含所有测试样本的特征
        score_bank 包含所有测试样本的分类输出
        label_bank 包含所有测试样本的真实标签。'''
        # 据 softmax 输出，找到每个样本最可能的类别
        _, old_predictions = torch.max(score_bank, 1)
        old_predictions = old_predictions.cpu().numpy()

        # 对特征矩阵进行 L2 范数归一化，以便进行余弦相似度计算
        # 增加一个常数偏置维度可以避免某些情况下特征向量的范数为零，从而避免数值不稳定性问题。
        if args.distance == 'cosine':
            pass
            NP_fea_bank = torch.cat((fea_bank, torch.ones(fea_bank.size(0), 1)), 1)
            NP_fea_bank = (NP_fea_bank.t() / torch.norm(NP_fea_bank, p=2, dim=1)).t()

        # 计算分类的类别数
        K = score_bank.size(1)

        ## 整体转换回NumPy
        # 将特征矩阵移回 CPU 并转换为 NumPy 数组
        NP_fea_bank = NP_fea_bank.float().cpu().numpy()
        # 将特征矩阵移回 CPU 并转换为 NumPy 数组
        NP_score_bank = score_bank.float().cpu().numpy()
        # 将特征矩阵移回 CPU 并转换为 NumPy 数组
        NP_label_bank = label_bank.numpy()

        aff_bank = NP_score_bank
        '''迭代循环，进行了原型生成和伪标签更新的过程，这是基于原型和距离计算的类别估计'''
        for _ in range(2):
            # 计算原型的初始化位置，通过将分类概率与特征矩阵相乘
            initc = aff_bank.T.dot(NP_fea_bank) / (1e-8 + aff_bank.sum(axis=0)[:, None])
            # 计算所有样本与初始化原型之间的距离，使用指定的距离度量方式（余弦相似度）
            dd = cdist(NP_fea_bank, initc, args.distance)

            # 根据每个样本调整后的距离的最小值来更新伪标签
            now_predictions = dd.argmin(axis=1)
            min_distances = dd.min(axis=1)

            # 找到高于阈值的样本索引，越大越不相似
            close_samples_indices = np.where(min_distances > args.threshold)[0]
            # 将伪标签转换为一个one-hot形式的矩阵，用于下一轮迭代
            aff_bank = np.eye(K)[now_predictions]
            aff_bank[close_samples_indices] = NP_score_bank[close_samples_indices]

        # 将选择困难症样本以-1的方式保存到predict中
        now_predictions[close_samples_indices] = -1
        prediction_bank[i] = now_predictions

    result = tools.check_columns(prediction_bank)
    class_range = np.arange(-1, args.class_num)
    prediction_counts = np.bincount(
        np.searchsorted(class_range, prediction_bank[-1], sorter=np.argsort(class_range)),
        minlength=len(class_range),
    )
    result_counts = np.bincount(
        np.searchsorted(class_range, result, sorter=np.argsort(class_range)),
        minlength=len(class_range),
    )

    # 是否需要数据增强增添样本
    # 否
    if args.mix == 1:
        add_loader, add_result, add_fea_bank, add_score_bank, add_log = loader, result, fea_bank, score_bank, ""
    # 是
    elif args.mix == 2:
        add_loader, add_result, add_fea_bank, add_score_bank, add_log = text.kk(args, loader, result, networks,
                                                                                fea_bank,
                                                                                score_bank)
    else:
        a = b

    # 计算正确率（默认过滤）
    pseudo_acc = np.mean(result[result != -1] == NP_label_bank[result != -1])
    nonpseudo_acc = np.mean((old_predictions[result != -1] == NP_label_bank[result != -1]))
    _, now_matrix = tools.cal_acc_matrix(NP_label_bank[result != -1], result[result != -1])
    _, now_print_matrix = tools.print_acc(now_matrix)

    # 过滤掉的标签的正确率
    filter_nonpseudo_acc = np.mean((old_predictions[result == -1] == NP_label_bank[result == -1]))
    _, filter_matrix = tools.cal_acc_matrix(NP_label_bank[result == -1], old_predictions[result == -1])
    _, now_filter_matrix = tools.print_acc(filter_matrix)

    # 保存在日志中
    log_str = f'已过滤\t未用伪标签：{nonpseudo_acc * 100:.2f}% -> 使用伪标签：{pseudo_acc * 100:.2f}%\n' \
              f'投票前-排除样本：{prediction_counts[0]}|{len(fea_bank)}\t通过率:{(1-(prediction_counts[0]/len(fea_bank)))*100:.2f}%\t剩余：' + ', '.join(map(str, prediction_counts[1:])) + "\n" \
              f'投票后-排除样本：{result_counts[0]}|{len(fea_bank)}\t通过率:{(1-(result_counts[0]/len(fea_bank)))*100:.2f}%\t剩余：' + ', '.join(map(str, result_counts[1:])) + "\n"
    log_str += add_log
    log_str += now_print_matrix
    log_str += f'\n被排除样本 正确率：{filter_nonpseudo_acc * 100:.2f}%\n'
    log_str += now_filter_matrix
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return add_result.astype('int'), add_loader, add_fea_bank, add_score_bank


def train_target(args, dset_loaders, networks):
    ## 网络初始化
    global fea_bank, label_bank, score_bank

    ## 参数设置
    ## 优化器设置
    param_group = []  # 创建一个空列表，用于存储网络模型中的参数组信息
    learning_rate = args.lr  # 从超参数中获取学习率
    '''
    遍历网络的所有参数
    将每个参数及其对应的学习率添加到 param_group 列表中'''
    for idx, net in enumerate(networks):
        for k, v in net.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate * args.lr_f[idx]}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    for net in networks:
        net.eval()

    ## 开始训练
    # 训练参数初始化
    initial_threshold = args.threshold  # 初始化args.threshold
    max_iter = args.max_epoch * len(dset_loaders["train"])  # 最大训练批次
    interval_iter = max_iter // args.interval  # 诊断间隔
    iter_num = 0  # 当前批次数
    acc_init = 0  # 正确率记录
    start = time.time()  # 开始时间设置
    real_max_iter = max_iter
    mix_train = tools.mix_train()

    # 跑前看看正确率
    acc_t_te, acc_matrix, _ = tools.cal_acc(dset_loaders["val"], networks)
    log_str = "任务: {}; 批次:{:·>5d}/{:·>5d};  总正确率: {:4.2f}%". \
                  format(args.name, iter_num, max_iter, acc_t_te) + "\n"
    correct_rate, acc_str = tools.print_acc(acc_matrix)
    log_str += acc_str

    args.out_file.write(log_str + "\n")
    args.out_file.flush()

    print(log_str + "\n")

    for net in networks:
        net.train()

    # 迭代训练
    while iter_num < real_max_iter:
        # 检查是否到了伪标签生成(更新)的时间点
        #### 实验：检查是否到了更新仓库数据的时间点
        # 通过伪标签权重 args.cls_par 判断是否使用伪标签，如果 cls_par > 0，则使用伪标签
        if iter_num % interval_iter == 0:
            for net in networks:
                net.eval()
            mem_label, loader, fea_bank, score_bank = obtain_label(dset_loaders["train"], networks, args)
            # 将生成的伪标签从NumPy数组转换为PyTorch张量，并将其移动到GPU上
            mem_label = torch.from_numpy(mem_label).cuda()
            # 将生成伪标签的数据与“标签”再次丢入DataLoader中，借助伪标签进行后续训练
            for net in networks:
                net.train()
            # 由于数据集发生变动，因此直接重新设置迭代器
            iter_test = iter(loader)

        try:
            # 从目标域数据集中获取一个 batch 的数据
            inputs_test, _, tar_idx = next(iter_test)
        except:
            # 如果迭代器耗尽，则重新获取新的迭代器
            iter_test = iter(loader)
            inputs_test, _, tar_idx = next(iter_test)

        # -1是选择困难症样本，在伪标签损失中应该剔除掉
        # 制作掩码列表，防止选择困难样本进入伪标签损失
        non_mask = mem_label[tar_idx] != -1

        # 忽略包含单个样本的 batch 或 全-1样本
        if inputs_test.size(0) == 1 or not torch.any(non_mask):
            if not torch.any(non_mask):
                iter_num += 1
            continue

        # 将输入数据移至 GPU
        inputs_test = inputs_test.cuda()
        # 更新迭代次数
        iter_num += 1

        # 调整优化器的学习率
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        # 预测值
        labels_test = mem_label[tar_idx]
        # 梯度清零
        optimizer.zero_grad()

        '''____________________________________mix伪标签损失____________________________________'''
        pk = args.mix
        args.mix = 0
        mix_loss = mix_train(args, networks, inputs_test[non_mask], labels_test[non_mask])
        args.mix = pk
        '''____________________________________经过模型____________________________________'''
        features_test = inputs_test
        for net in networks[:-1]:
            features_test = net(features_test)
        # 分类器 用于对处理后的特征进行分类，得到源域的分类结果
        outputs_test = networks[-1](features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        '''____________________________________伪标签损失计算____________________________________'''
        # args.cls_par 表示使用伪标签的权重，如果大于零则使用伪标签
        if args.cls_par > 0 and non_mask.any():
            # 使用交叉熵损失计算分类器损失
            classifier_loss = loss.CrossEntropyLabelSmooth(
                num_classes=args.class_num,
                epsilon=args.smooth,
                reduction=False
            )(outputs_test[non_mask], labels_test[non_mask]) * args.cls_par
            # 将分类器损失乘以权重 args.cls_par
            classifier_loss *= args.cls_par

        else:  # 如果不使用伪标签，则将分类器损失设置为零
            classifier_loss = torch.tensor([0.0]).cuda()

        '''____________________________________伪标签(包含Mix和原始)损失____________________________________'''
        combined_loss = torch.mean(
            torch.cat([loss for loss in [mix_loss, classifier_loss] if torch.mean(loss) > 0])
        )
        classifier_loss = torch.mean(classifier_loss)

        '''____________________________________熵损失____________________________________'''
        # 检查非掩码样本是否存在
        if (~non_mask).any() and args.ent_par != -1:
            entropy_non_mask = loss.Entropy(softmax_out[~non_mask]).mean()
        else:
            entropy_non_mask = 0.0  # 如果无有效非掩码样本，设置为 0

        if non_mask.any():
            entropy_mask = loss.Entropy(softmax_out[non_mask]).mean()
            softmax_mean = torch.clamp(softmax_out[non_mask].mean(dim=0), min=args.epsilon)
            softmax_log_term = (-softmax_mean * torch.log(softmax_mean + args.epsilon)).sum()
        else:
            entropy_mask = 0.0
            softmax_log_term = 0.0

        # 计算最终损失
        entropy_loss = entropy_mask - entropy_non_mask - softmax_log_term

        '''____________________________________add损失______________________________________'''
        # 排除标签为 -1 的样本
        valid_mask = labels_test != -1  # 筛选有效样本
        valid_features = features_test[valid_mask]  # 有效样本的特征
        valid_labels = labels_test[valid_mask]  # 有效样本的标签
        valid_softmax = softmax_out[valid_mask]

        if non_mask.any() and args.aad_par != 0:
            # 正则化有效样本的特征
            output_features = F.normalize(valid_features)

            # # 正则化有效样本的特征
            # output_features = valid_softmax

            # **1. 同标签样本损失（正样本损失）**
            # 构建同标签样本的掩码
            same_label_mask = (valid_labels.unsqueeze(0) == valid_labels.unsqueeze(1)).float()  # valid_batch x valid_batch
            same_label_mask.fill_diagonal_(0)  # 去掉对角线上的自身样本

            # 筛选同标签的样本对特征
            same_label_features = output_features.unsqueeze(0) * same_label_mask.unsqueeze(
                -1)  # valid_batch x valid_batch x feature_dim
            same_label_similarities = (same_label_features @ output_features.unsqueeze(-1)).squeeze(
                -1)  # valid_batch x valid_batch

            # 每行求和以计算同标签的相似度
            row_sums = same_label_similarities.sum(dim=1)

            # 保留非零元素，并计算正样本损失（feature_loss）
            non_zero_sums = row_sums[row_sums != 0]
            feature_loss = -non_zero_sums.mean() if non_zero_sums.numel() > 0 else torch.tensor(0.0, device=same_label_similarities.device)

            # **2. 不同标签样本损失（负样本损失）**
            # 构建不同标签的掩码
            neg_mask = 1 - same_label_mask
            neg_mask.fill_diagonal_(0)  # 自身样本不作为负样本

            # 筛选不同标签的样本对特征
            neg_label_features = output_features.unsqueeze(0) * neg_mask.unsqueeze(
                -1)  # valid_batch x valid_batch x feature_dim
            neg_label_similarities = (neg_label_features @ output_features.unsqueeze(-1)).squeeze(
                -1)  # valid_batch x valid_batch

            # 筛选不同标签样本的相似度并按行求和
            row_sums_neg = neg_label_similarities.sum(dim=1)  # 每行求和

            # 保留非零元素并计算负样本损失
            non_zero_neg_sums = row_sums_neg[row_sums_neg != 0]
            neg_pred_loss = non_zero_neg_sums.mean() if non_zero_neg_sums.numel() > 0 else torch.tensor(0.0,
                                                                                                        device=output_features.device)

            # **最终 add 损失**
            add_loss = (feature_loss + args.K * neg_pred_loss) * args.aad_par

        else:
            add_loss = 0.0

        # 打印情况
        if ((iter_num) % len(dset_loaders["train"])) == 0:
            print(
                "轮次:{:·>3d}/{:·>3d} || c_Loss:{:4.3f} || e_Loss:{:4.3f} || Aad_Loss:{:4.3f} || 耗时：{:4.2f}s".format(
                    iter_num // len(dset_loaders["train"]),
                    args.max_epoch,
                    combined_loss,
                    entropy_loss,
                    add_loss,
                    time.time() - start))
            start = time.time()

        final_loss = combined_loss + add_loss + entropy_loss

        # 反向传播
        final_loss.backward()
        optimizer.step()

        ## 正确率计算并且保存最佳模型
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for net in networks:
                net.eval()

            acc_t_te, acc_matrix, _ = tools.cal_acc(dset_loaders["val"], networks)
            log_str = "任务: {}; mix批次:{:·>5d}; 批次:{:·>5d}/{:·>5d};  总正确率: {:4.2f}%". \
                          format(args.name, mix_train.mix_iter_num, iter_num, max_iter, acc_t_te) + "\n"
            # log_str = "任务: {};  批次:{:·>5d}/{:·>5d};  总正确率: {:4.2f}%". \
            #               format(args.name, iter_num, max_iter, acc_t_te) + "\n"

            correct_rate, acc_str = tools.print_acc(acc_matrix)
            log_str += acc_str

            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str + "\n")

            for net in networks:
                net.train()

            if acc_t_te > acc_init:
                ## 更新正确率
                acc_init = acc_t_te

                ## 保存模型
                for idx, net in enumerate(networks[:-1], start=1):
                    best_net = net.state_dict()
                    torch.save(best_net, osp.join(args.output_tar_logpt, f"source_{idx}.pt"))
                best_netC = networks[-1].state_dict()
                torch.save(best_netC, osp.join(args.output_tar_logpt, "source_C.pt"))

        if iter_num == max_iter:
            for idx, net in enumerate(networks[:-1], start=1):
                net.load_state_dict(torch.load(f"{args.output_tar_logpt}/source_{idx}.pt"))
            networks[-1].load_state_dict(torch.load(f"{args.output_tar_logpt}/source_C.pt"))

            for net in networks:
                net.eval()

            acc_t_te, acc_matrix, _ = tools.cal_acc(dset_loaders["test"], networks)
            log_str = '测试集任务: {}; 总正确率 = {:4.2f}%'.format(args.name, acc_t_te) + '\n'
            correct_rate, acc_str = tools.print_acc(acc_matrix)
            log_str += acc_str

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            # 将测试集成绩记录在案
            args.class_acc_list = [np.mean(correct_rate)] + correct_rate
            args.acc_list[args.find_k(args.s, args.t)] = acc_t_te

            for net in networks:
                net.train()

    return networks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPA")
    # 选择GPU设备
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    # 选择源域
    parser.add_argument("--s", type=int, default=-1, help="source")
    # 选择目标域
    parser.add_argument("--t", type=int, default=-1, help="target")
    # 并行工作数
    parser.add_argument("--worker", type=int, default=0, help="number of workers")
    # 随机种子
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    # 当数据为0时，代替数值
    parser.add_argument("--epsilon", type=float, default=1e-5)
    # 标签平滑参数
    parser.add_argument('--smooth', type=float, default=0.1)
    # 同类型样本数量限制(-1为无限制)
    parser.add_argument('--num', type=int, default=-1)
    # 训练策略
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    # 特征距离选择
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    #
    parser.add_argument("--issave", type=bool, default=True)

    parser.add_argument("--cc", default=False, action="store_true")
    # 阿尔法参数
    parser.add_argument("--alpha", type=float, default=1.0)
    # 贝塔参数
    parser.add_argument("--beta", type=float, default=5.0)
    # 阿尔法参数衰减
    parser.add_argument("--alpha_decay", default=True)

    parser.add_argument("--nuclear", default=False, action="store_true")

    parser.add_argument("--var", default=False, action="store_true")

    parser.add_argument("--choose", default=True)

    # 神经网络（此超参数已被架空）
    parser.add_argument("--net", type=str, default="resnet18")

    # 训练轮次
    parser.add_argument("--max_epoch", type=int, default=20, help="max iterations")
    # 计算正确率次数，你写多少，这次训练总共就会生成多少次正确率计算
    parser.add_argument("--interval", type=int, default=4)
    # 训练批次
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    # 学习率
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    # 学习率倍率（仅影响特征提取）
    parser.add_argument('--lr_f', type=str, default="[0.1, 0.1, 1]")
    # 进行Mixup的概率
    parser.add_argument('--mix', type=float, default=2)
    # 伪标签样本阈值判断(趋于2筛选越宽容；趋于0筛选越严格)
    parser.add_argument('--threshold', type=float, default=0.4)
    # 伪标签权重
    parser.add_argument('--cls_par', type=float, default=1)
    # 熵权重
    parser.add_argument('--ent_par', type=float, default=1)
    # AaD权重
    parser.add_argument('--aad_par', type=float, default=1)
    # 近邻样本数量
    parser.add_argument("--K", type=float, default=0.6)
    # 噪音干扰倍率
    parser.add_argument('--noise_std', type=float, default=0)
    # dropout概率
    parser.add_argument('--dropout', type=float, default=0.1)

    # 中介器选择
    parser.add_argument('--layer', type=str, default="bn", choices=["ori", "bn", "None"])
    # 中介输出的维度(中介器为None时无视这个参数)
    parser.add_argument('--bottleneck', type=int, default=256)
    # 分类器选择
    parser.add_argument('--classifier', type=str, default="wn", choices=["linear", "wn", "MLP"])

    # 输出文件夹
    parser.add_argument("--output", type=str, default="weight")

    # 数据集向导
    parser.add_argument('--username', type=str, default="WWY_JUN_RES")  # 一般定死
    parser.add_argument('--data_name', type=str, default="JNU_1d_2048_2000")
    parser.add_argument('--domain_names', type=list, default=['600', '800', '1000'])  # 一般定死
    # ['N15_M01_F10', 'N15_M07_F10', 'N15_M07_F04']
    parser.add_argument('--class_num', type=int, default=4)  # 一般定死
    parser.add_argument('--folder_root', type=str, default="./DATA")  # 一般定死

    args = parser.parse_args()

    args.choose_dict = ["max_epoch", "interval", "batch_size", "lr", "lr_f",
                        "mix", "threshold", "cls_par", 'ent_par', 'aad_par', "K", "noise_std", "dropout", "classifier"]

    # 设立随机数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # 获取当前日期和时间
    args.nowtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    len_names = len(args.domain_names)

    # 修改学习率倍率格式
    args.lr_f = json.loads(args.lr_f)

    # 获得文件完整路径
    args.data_folder = args.input_src_logpt = osp.join(
        args.folder_root,
        args.data_name
    )

    # 选择
    if args.s >= len_names or args.t >= len_names:
        raise ValueError(f'源域或目标域编码出界，编码不能超过{len_names - 1}')
    else:
        source_mem = args.s
        target_mem = args.t
        args.acc_list = [-1] * (len_names ** 2 - len_names)
        args.find_k = find_position(len_names)
        for args.s in range(len_names) if source_mem == -1 else [source_mem]:
            # pt存放位置
            args.input_src_logpt = osp.join(
                args.output,
                args.username,
                "Source-源域",
                "log&pt",
                "(B{})".format(args.s + 1) + args.domain_names[args.s].upper()
            )
            print("读取源域路径：{}".format(args.input_src_logpt))

            # excel输出位置（所有excel，包括总结excel和不同领域的细分excel）
            args.output_tar_excel = osp.join(
                args.output,
                args.username,
                "Target-目标域",
            )
            print("excel输出位置：{}".format(args.output_tar_excel))

            # 特征图片输出位置
            args.output_tar_feapic = osp.join(
                args.output,
                args.username,
                "Picture-特征图",
                datetime.strptime(args.nowtime, "%Y-%m-%d %H:%M:%S").strftime("%Y年%m月%d日%H时%M分"),
                "(B{})".format(args.s + 1) + args.domain_names[args.s].upper()
            )
            print("特征图片输出位置：{}".format(args.output_tar_feapic))

            feature_dict = {}
            label_dict = {}
            score_dict = {}
            init_center_dict = {}

            args.names = ["Before" + ("(B{})".format(args.s + 1)) + args.domain_names[args.s].upper()]
            args.dset_paths = [osp.join(args.data_folder, args.domain_names[args.s] + '_label.txt')]

            args.input_dim = len(txt_loader(make_dataset(open(args.dset_paths[0]).readlines()[:1], None)[0][0]))
            source_networks = tools.read_model(args, model_path=args.input_src_logpt)

            # feature_out, label_out, score_out, init_center_out = plot_t_SNE.prepare_dicts(args, source_networks)
            # feature_dict.update(feature_out)
            # label_dict.update(label_out)
            # score_dict.update(score_out)
            # init_center_dict.update(init_center_out)

            for args.t in range(len_names) if target_mem == -1 else [target_mem]:
                if args.s == args.t: continue
                networks = source_networks
                args.class_acc_list = [-1] * (args.class_num + 1)

                print("现在选取:(B{}){} 进行训练的源模型".format(args.s + 1, args.domain_names[args.s]))
                print(
                    "现在进行：(B{}){}→(B{}){} 的迁移任务".format(args.s + 1, args.domain_names[args.s], args.t + 1,
                                                                 args.domain_names[args.t]))

                args.name = ("(B{})".format(args.s + 1)) + args.domain_names[args.s].upper() + "→" + (
                    "(B{})".format(args.t + 1)) + \
                            args.domain_names[args.t].upper()
                args.dset_path = osp.join(args.data_folder, args.domain_names[args.t] + '_label.txt')

                # log&pt输出位置（所有log&pt的输出位置，里面再细分不同的领域文件）
                args.output_tar_logpt = osp.join(
                    args.output_tar_excel,
                    "log&pt",
                    args.name
                )
                print("log&pt输出位置：{}".format(args.output_tar_logpt))

                # 获得迁移前的特征图
                args.names = ["Before" + args.name]
                args.dset_paths = [args.dset_path]
                # feature_out, label_out, score_out, init_center_out = plot_t_SNE.prepare_dicts(args, networks)
                # feature_dict.update(feature_out)
                # label_dict.update(label_out)
                # score_dict.update(score_out)
                # init_center_dict.update(init_center_out)

                # 创建文件夹
                # 直接创建log&pt的对应领域的文件夹，就会包含所有需要的路径了
                if not osp.exists(args.output_tar_logpt):
                    os.system("mkdir -p " + args.output_tar_logpt)
                if not osp.exists(args.output_tar_logpt):
                    os.makedirs(args.output_tar_logpt, exist_ok=True)

                args.out_file = open(
                    osp.join(args.output_tar_logpt,
                             "log_{}.txt".format(args.nowtime.replace("-", "").replace(":", "").replace(" ", "_"))), "w"
                )
                args.out_file.write(print_args(args) + "\n")
                args.out_file.flush()

                # 训练模型
                data_loaders = tools.args_data_load(args, {"train": 1, "val": 1, "test": 1}, separate=False)
                networks = train_target(args, data_loaders, networks)

                # 获得迁移之后的特征图
                args.names = ["After" + args.name]
                args.dset_paths = [args.dset_path]
                # feature_out, label_out, score_out, init_center_out = plot_t_SNE.prepare_dicts(args, networks)
                # feature_dict.update(feature_out)
                # label_dict.update(label_out)
                # score_dict.update(score_out)
                # init_center_dict.update(init_center_out)

                str_list = [f"B{args.s + 1}→B{args.t + 1}"]
                str_list.extend(['类型{}'.format(y + 1) for y in range(args.class_num)])
                write_to_excel(print_excel(args, str_list, args.class_acc_list),
                               args.output_tar_excel + f'/(B{args.s + 1})→(B{args.t + 1})tar_class_output.xlsx')

            # plot_t_SNE.plot_test_target(feature_dict, label_dict, score_dict, init_center_dict, args.output_tar_feapic)

        str_list = ['B{}→B{}'.format(x + 1, y + 1) for x in range(len_names) for y in range(len_names) if x != y]
        # 写入数据到 Excel 文件
        write_to_excel(print_excel(args, str_list, args.acc_list), args.output_tar_excel + '/tar_output.xlsx')
