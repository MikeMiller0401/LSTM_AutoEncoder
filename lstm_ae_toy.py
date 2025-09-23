import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.LSTMAE import LSTMAE
from train_utils import train_model, eval_model

parser = argparse.ArgumentParser(description='LSTM_AE TOY EXAMPLE')  # 建立一个命令行参数解析器，描述为 “LSTM_AE TOY EXAMPLE”
# 训练控制
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')  # 每个批次的数据量
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train')  # 总训练轮数，每一轮会遍历完整训练集一次。
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')  # 默认优化器
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')  # 学习率
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')  # 权重衰减（L2 正则化系数），用于防止过拟合
parser.add_argument('--grad-clipping', type=float, default=None, metavar='GC', help='gradient clipping value')  # 梯度裁剪阈值，避免梯度爆炸
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batch iteration to log status')  # 每多少个 batch 打印一次训练日志
# 模型结构
parser.add_argument('--hidden-size', type=int, default=256, metavar='N', help='LSTMAE hidden state size')  # LSTMAE 隐藏层维度，决定模型容量
parser.add_argument('--input-size', type=int, default=1, metavar='N', help='input size')  # LSTMAE 输入特征维度（比如单变量时间序列时是 1，多变量时可设大于 1）
parser.add_argument('--dropout', type=float, default=0.0, metavar='D', help='dropout ratio')  # dropout 比例，防止过拟合
parser.add_argument('--model-type', default='LSTMAE', help='currently only LSTMAE')  # 模型类型
parser.add_argument('--seq-len', default=50, help='sequence full size')  # 输入序列的时间步长度（即 LSTMAE 每次处理多少时间点）
# 输出与超参数
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')  # 模型 checkpoint 保存路径
parser.add_argument('--run-grid-search', action='store_true', default=False, help='Running hyper-parameters grid search')  # 是否运行超参数网格搜索

args = parser.parse_args(args=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# folder settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)


class toy_dataset(torch.utils.data.Dataset):
    def __init__(self, toy_data, labels):
        self.toy_data = toy_data
        self.labels = labels

    def __len__(self):
        return self.toy_data.shape[0]

    def __getitem__(self, index):
        return self.toy_data[index], self.labels[index]


def main():
    # Create data loaders with anomalies 构建包含异常数据的dataset，包含训练集 (train_iter)、验证集 (val_iter)、测试集 (test_iter)
    train_iter, val_iter, test_iter = create_dataloaders_with_anomalies(args.batch_size)

    # Create model 模型建立
    model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size,
                   dropout_ratio=args.dropout, seq_len=args.seq_len).to(device)


    optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=args.lr, weight_decay=args.wd)  # 动态选择优化器，例如 Adam / SGD，由 args.optim 决定
    criterion = torch.nn.MSELoss(reduction='sum') # 损失函数：采用 MSE (均方误差)，使用 sum 表示误差求和

    # Train
    # 在指定的 epoch 数内，依次执行训练和验证
    for epoch in range(args.epochs):
        train_model(criterion, epoch, model, args.model_type, optimizer, train_iter,
                    args.batch_size, args.grad_clipping, args.log_interval)  # 训练模型：计算损失并反向传播
        eval_model(criterion, model, args.model_type, val_iter)  # 验证模型：在验证集上评估模型性能

    # Test anomaly detection 异常检测 (测试阶段)
    # 在测试集上进行异常检测，返回：
    # errors   -> 重建误差值
    # labels   -> 真实标签 (正常/异常)
    # preds    -> 模型预测 (正常/异常)
    # threshold -> 区分正常/异常的阈值
    errors, labels, preds, threshold = detect_anomalies(model, test_iter, criterion)
    print(f"Detection threshold={threshold:.4f}")  # 打印检测阈值
    # 计算预测精度 (preds 与真实标签 labels 的一致率)
    acc = (preds == labels).mean()
    print(f"Anomaly detection accuracy: {acc:.4f}")

    # 绘制重建误差分布
    plot_anomaly_distribution(errors, labels, threshold)



def create_toy_data(num_of_sequences=10000, sequence_len=50) -> torch.tensor:
    """
    Generate num_of_sequences random sequences with length of sequence_len each.
    :param num_of_sequences: number of sequences to generate
    :param sequence_len: length of each sequence
    :return: pytorch tensor containing the sequences
    """
    # Random uniform distribution
    toy_data = torch.rand((num_of_sequences, sequence_len, 1))

    return toy_data


def create_toy_data_with_anomalies(num_of_sequences=10000, sequence_len=50, anomaly_ratio=0.05):
    """
    生成带有异常的 toy 数据集
    :param num_of_sequences: 总序列数
    :param sequence_len: 每个序列的长度
    :param anomaly_ratio: 异常比例
    :return: (toy_data, labels) 其中 labels=0 正常，1 异常
    """
    # 正常数据（均匀分布）
    toy_data = torch.rand((num_of_sequences, sequence_len, 1))
    labels = torch.zeros(num_of_sequences)  # 默认全是正常

    # 随机挑一些样本注入异常
    num_anomalies = int(num_of_sequences * anomaly_ratio)
    anomaly_indices = np.random.choice(num_of_sequences, num_anomalies, replace=False)

    for idx in anomaly_indices:
        # 复制正常序列
        seq = toy_data[idx]

        # 异常类型 1: 突变 spike
        if np.random.rand() > 0.5:
            pos = np.random.randint(0, sequence_len)
            seq[pos:] += 2.0  # 突然升高

        # 异常类型 2: 整体偏移 shift
        else:
            seq += 1.5  # 整个序列整体偏移

        toy_data[idx] = seq
        labels[idx] = 1  # 标记异常

    return toy_data, labels


def create_dataloaders(batch_size, train_ratio=0.6, val_ratio=0.2):
    """
    Build train, validation and tests dataloader using the toy data
    :return: Train, validation and test data loaders
    """
    toy_data = create_toy_data()
    len = toy_data.shape[0]

    train_data = toy_data[:int(len * train_ratio), :]
    val_data = toy_data[int(train_ratio * len):int(len * (train_ratio + val_ratio)), :]
    test_data = toy_data[int((train_ratio + val_ratio) * len):, :]

    print(f'Datasets shapes: Train={train_data.shape}; Validation={val_data.shape}; Test={test_data.shape}')
    train_iter = torch.utils.data.DataLoader(toy_dataset(train_data), batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(toy_dataset(val_data), batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(toy_dataset(test_data), batch_size=batch_size, shuffle=False)

    return train_iter, val_iter, test_iter

def create_dataloaders_with_anomalies(batch_size, train_ratio=0.6, val_ratio=0.2):
    toy_data, labels = create_toy_data_with_anomalies()
    length = toy_data.shape[0]

    # 按比例划分
    train_data = toy_data[:int(length * train_ratio)]
    train_labels = labels[:int(length * train_ratio)]

    val_data = toy_data[int(length * train_ratio):int(length * (train_ratio + val_ratio))]
    val_labels = labels[int(length * train_ratio):int(length * (train_ratio + val_ratio))]

    test_data = toy_data[int(length * (train_ratio + val_ratio)):]
    test_labels = labels[int(length * (train_ratio + val_ratio)):]

    print(f"Datasets shapes: Train={train_data.shape}; Validation={val_data.shape}; Test={test_data.shape}")

    train_iter = torch.utils.data.DataLoader(toy_dataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(toy_dataset(val_data, val_labels), batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(toy_dataset(test_data, test_labels), batch_size=batch_size, shuffle=False)

    return train_iter, val_iter, test_iter

def detect_anomalies(model, test_iter, criterion, threshold=None):
    model.eval()
    errors, labels = [], []
    with torch.no_grad():
        for seq, lbl in test_iter:
            seq = seq.to(device)
            rec = model(seq)

            # 对每个样本单独计算重建误差 (MSE)
            batch_errors = torch.mean((rec - seq) ** 2, dim=(1, 2))  # shape: [batch_size]
            errors.extend(batch_errors.cpu().numpy())
            labels.extend(lbl.cpu().numpy())

    errors = np.array(errors)
    labels = np.array(labels)

    # 如果没给阈值，就用均值+3倍标准差
    if threshold is None:
        threshold = errors.mean() + 3 * errors.std()

    preds = (errors > threshold).astype(int)  # 预测结果：1=异常
    return errors, labels, preds, threshold

def plot_anomaly_distribution(errors, labels, threshold):
    """
    绘制散点图：样本序号 vs 重建误差
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(errors)), errors, c=labels, cmap="coolwarm", alpha=0.7, label="Samples")
    plt.axhline(threshold, color="b", linestyle="--", linewidth=2, label=f"Threshold={threshold:.2f}")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.title("Anomaly Detection Results")
    plt.legend()
    plt.show()



def plot_toy_data(toy_example, description, color='b'):
    """
    Recieves a toy raw data sequence and plot it
    :param toy_example: toy data example sequence
    :param description: additional description to the plot
    :param color: graph color
    :return:
    """
    time_lst = [t for t in range(toy_example.shape[0])]

    plt.figure()
    plt.plot(time_lst, toy_example.tolist(), color=color)
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    # plt.legend()
    plt.title(f'Single value vs. time for toy example {description}')
    plt.show()


def plot_orig_vs_reconstructed(model, test_iter, num_to_plot=2):
    """
    Plot the reconstructed vs. Original MNIST figures
    :param model: model trained to reconstruct MNIST figures
    :param test_iter: test data loader
    :param num_to_plot: number of random plots to present
    :return:
    """
    model.eval()
    # Plot original and reconstructed toy data
    plot_test_iter = iter(torch.utils.data.DataLoader(test_iter.dataset, batch_size=1, shuffle=False))

    for i in range(num_to_plot):
        orig, _ = next(plot_test_iter)
        orig = orig.to(device)
        with torch.no_grad():
            rec = model(orig)

        time_lst = [t for t in range(orig.shape[1])]

        # Plot original
        plot_toy_data(orig.squeeze(), f'Original sequence #{i + 1}', color='g')

        # Plot reconstruction
        plot_toy_data(rec.squeeze(), f'Reconstructed sequence #{i + 1}', color='r')

        # Plot combined
        plt.figure()
        plt.plot(time_lst, orig.squeeze().tolist(), color='g', label='Original signal')
        plt.plot(time_lst, rec.squeeze().tolist(), color='r', label='Reconstructed signal')
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.legend()
        title = f'Original and Reconstruction of Single values vs. time for toy example #{i + 1}'
        plt.title(title)
        plt.savefig(f'{title}.png')
        plt.show()


def hyper_params_grid_search(train_iter, val_iter, criterion):
    """
    Function to perform hyper-parameter grid search on a pre-defined range of values.
    :param train_iter: train dataloader
    :param val_iter: validation data loader
    :param criterion: loss criterion to use (MSE for reconstruction)
    :return:
    """
    lr_lst = [1e-2, 1e-3, 1e-4]
    hs_lst = [16, 32, 64, 128, 256]
    clip_lst = [None, 10, 1]

    total_comb = len(lr_lst) * len(hs_lst) * len(clip_lst)
    print(f'Total number of combinations: {total_comb}')

    curr_iter = 1
    best_param = {'lr': None, 'hs': None, 'clip_val': None}
    best_val_loss = np.Inf
    params_loss_dict = {}

    for lr in lr_lst:
        for hs in hs_lst:
            for clip_val in clip_lst:
                print(f'Starting Iteration #{curr_iter}/{total_comb}')
                curr_iter += 1
                model = LSTMAE(input_size=args.input_size, hidden_size=hs, dropout_ratio=args.dropout,
                               seq_len=args.seq_len)
                model = model.to(device)
                optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=lr, weight_decay=args.wd)

                for epoch in range(args.epochs):
                    # Train loop
                    train_model(criterion, epoch, model, args.model_type, optimizer, train_iter, args.batch_size, clip_val,
                                args.log_interval)
                avg_val_loss, val_acc = eval_model(criterion, model, args.model_type, val_iter)
                params_loss_dict.update({f'lr={lr}_hs={hs}_clip={clip_val}': avg_val_loss})
                if avg_val_loss < best_val_loss:
                    print(f'Found better validation loss: old={best_val_loss}, new={avg_val_loss}; parameters: lr={lr},hs={hs},clip_val={clip_val}')
                    best_val_loss = avg_val_loss
                    best_param = {'lr': lr, 'hs': hs, 'clip_val': clip_val}

    print(f'Best parameters found: {best_param}')
    print(f'Best Validation Loss: {best_val_loss}')
    print(f'Parameters loss: {params_loss_dict}')



if __name__ == '__main__':
    main()
