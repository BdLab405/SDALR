import torch
import numpy as np
import random

from torch.utils.data import DataLoader

import tools
import torch.nn as nn
import torch.nn.functional as F
import data_list


def kk(args, loader, result, networks, fea_bank, score_bank):
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    # 创建一个足够大的空张量来存储所有数据
    datas = torch.empty((len(loader.dataset),) + loader.dataset[0][0].shape, dtype=torch.float32)
    # 遍历 DataLoader
    for data, _, idx in iter(loader):
        datas[idx] = data

    datas = torch.stack([data[0] for data in loader.dataset])
    counts_dict = dict(zip(*np.unique(result, return_counts=True)))
    counts_dict.pop(-1, None)
    counts_dict = {key: max(counts_dict.values()) - value for key, value in counts_dict.items()}

    # 初始化一个空字典
    grouped_data = {}

    # 遍历 result 和 datas，将数据按标签分组
    for data, label in zip(datas, result):
        if label not in grouped_data:
            grouped_data[label] = []  # 创建新的列表
        grouped_data[label].append(data)  # 将数据添加到对应的列表中
    grouped_data.pop(-1, None)

    # 假设处理类型列表
    process_types = ['reverse', 'shift']  # 可以根据需要添加更多处理类型

    # 新字典存放处理后的样本
    processed_data = {}
    processed_label = {}

    # 遍历每个标签
    for label, count in counts_dict.items():
        if label in grouped_data:  # 确保标签在 grouped_data 中存在
            # 随机选择 count 个样本
            selected_samples = random.sample(grouped_data[label], min(count, len(grouped_data[label])))

            # 为每个样本随机选择处理类型
            for sample in selected_samples:
                process_type = random.choice(process_types)
                if process_type not in processed_data:
                    processed_data[process_type] = []  # 创建新的列表
                    processed_label[process_type] = []
                processed_data[process_type].append(sample)
                processed_label[process_type].append(label)


    # 逐批次处理样本
    new_data = []
    new_label = []
    for process_type, samples in processed_data.items():
        # 调用 process_inputs 函数处理当前处理类型的所有样本
        new_data.append(tools.process_inputs(torch.stack(samples), process_type))
        new_label.extend(processed_label[process_type])

    if new_data:  # 如果 new_data 不为空
        new_data = torch.cat(new_data, dim=0)

        # 遍历数据，按批次输入模型
        with torch.no_grad():
            for i in range(0, new_data.shape[0], 32):
                # 获取当前批次的样本
                inputs = new_data[i:i + 32].cuda()  # 取到最后不够 a 个时也要取

                # 输入模型
                # 将输入数据传递到 netF 进行特征提取，然后通过 netB 得到特征
                features = inputs
                for net in networks[:-1]:
                    features = net(features)

                # 对特征进行归一化
                output_norm = F.normalize(features)
                # 使用 Softmax 得到分类概率
                outputs = nn.Softmax(-1)(networks[-1](features))

                try:
                    new_fea_bank = torch.cat([new_fea_bank, output_norm.detach().clone().cpu()], dim=0)
                    new_score_bank = torch.cat([new_score_bank, outputs.detach().clone()])
                except NameError:
                    new_fea_bank = output_norm.detach().clone().cpu()
                    new_score_bank = outputs.detach().clone()

        _, predictions = torch.max(new_score_bank, 1)
        acc = torch.mean((predictions == torch.tensor(new_label).cuda()).float())
        log = f'增添样本：{sum(counts_dict.values())}\t\t细则：' + ', '.join(
            [str(value) for key, value in sorted(counts_dict.items())]) + f'\t正确率估算：{acc * 100:.2f}%' + "\n"
        fea_bank = torch.cat([fea_bank, new_fea_bank], dim=0)
        score_bank = torch.cat([score_bank, new_score_bank])
        result = np.concatenate((result, np.array(new_label)))
        datas = torch.cat([datas, new_data], dim=0)

    else:
        log = f'增添样本：{sum(counts_dict.values())}\t\t细则：' + ', '.join(
            [str(value) for key, value in sorted(counts_dict.items())])

    dataset = data_list.ClearDataList_idx(datas, result)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False
    )

    return loader, result, fea_bank, score_bank, log
    pass

