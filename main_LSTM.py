# import mymodule
import sys
import numpy as np


import matplotlib.pyplot as plt
# from PyQt5 import QtWidgets
from sklearn import svm
# import scipy.io as sio


from LSTM.Lib_LSTM import *
import readData as rd
from pathlib import Path


def lstm_task():
    # # 设置随机种子，保证结果可复现
    # torch.manual_seed(42)
    # np.random.seed(42)
    #
    # print(torch.__version__)
    #
    # # 生成数据
    # seq_length = 10  # 序列长度
    # num_samples = 1000  # 总样本数
    # X, y, raw_data = generate_time_series_data(seq_length, num_samples)
    #
    # # 划分训练集和测试集
    # train_size = int(0.8 * len(X))
    # X_train, X_test = X[:train_size], X[train_size:]
    # y_train, y_test = y[:train_size], y[train_size:]
    #
    # sio.savemat("D:/testWork/python_module/LSTM/LSTM_data.mat", {'x': X_train, 'y': y_train, 'raw_data': raw_data})
    # sio.savemat("D:/testWork/python_module/LSTM/LSTM_data_test.mat", {'x': X_test, 'y': y_test})

    current_dir = Path.cwd()

    data_str = current_dir.joinpath("LSTM/LSTM_data.mat")
    xy_data = rd.read_data(data_str)
    print("xy_data")

    # data_str = "D:/testWork/python_module/LSTM/LSTM_data_test.mat"
    data_str = current_dir.joinpath("LSTM/LSTM_data_test.mat")
    xy_test_data = rd.read_data(data_str)
    print("xy_test_data")

    # config_str = "D:/testWork/python_module/LSTM/LSTM.yaml"
    config_str = current_dir.joinpath("LSTM/LSTM.yaml")
    config = rd.read_config(config_str)
    print("config")
    print(config)

    print_model()
    print("create_model")
    create_model(config)
    print_model()

    xy_data_list = [xy_data['x'], xy_data['y']]
    xy_test_data_list = [xy_test_data['x'], xy_test_data['y']]
    losses = fit(xy_data_list, xy_test_data_list)
    # save("ocsvm")

    # save('D:/testWork/python_module/LSTM/LSTM_model.pth')
    save(current_dir.joinpath("LSTM/LSTM_model.pth"))
    print_model()
    Lib_LSTM.model = None
    print_model()
    load(current_dir.joinpath("LSTM/LSTM_model.pth"))
    print_model()

    train_preds = predict(xy_data['x'])
    test_preds = predict(xy_test_data['x'])

    # 添加以下代码设置中文字体
    # plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]  # 支持中文的字体列表
    plt.rcParams["font.family"] = ["SimHei"]  # 支持中文的字体列表
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    # 6. 可视化结果
    plt.figure(figsize=(15, 8))

    train_size = xy_data['x'].shape[0]
    seq_length = xy_data['x'].shape[1]
    raw_data = np.squeeze(xy_data['raw_data'])
    # 绘制原始数据
    plt.subplot(2, 1, 1)
    plt.plot(raw_data, label='原始数据')
    plt.axvline(x=train_size + seq_length, color='r', linestyle='--', label='训练集/测试集分割线')
    plt.title('原始时序数据')
    plt.legend()

    # 绘制预测结果
    plt.subplot(2, 1, 2)
    plt.plot(range(seq_length, seq_length + len(train_preds)), train_preds, label='训练集预测')
    plt.plot(range(seq_length + len(train_preds), seq_length + len(train_preds) + len(test_preds)),
             test_preds, label='测试集预测', color='g')
    plt.plot(raw_data, label='原始数据', alpha=0.3)
    plt.axvline(x=train_size + seq_length, color='r', linestyle='--')
    plt.title('LSTM预测结果')
    plt.legend()

    plt.tight_layout()
    plt.show()

    train_losses = losses[0]
    test_losses = losses[1]
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.title('训练与测试损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    lstm_task()