from LSTMAE.Lib_LSTMAE import *
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
import readData as rd
def lstmae_task():
    print("LSTM-AE Begin.")
    current_dir = str(Path.cwd())

    # 建立&保存 数据集
    train_iter, val_iter, test_iter, train_data, train_labels, val_data, val_labels, test_data, test_labels = create_dataloaders_with_anomalies(batch_size=128)
    sio.savemat(current_dir + "/LSTMAE/LSTMAE_data_train.mat", {'x': train_data, 'y': train_labels})
    sio.savemat(current_dir + "/LSTMAE/LSTMAE_data_val.mat", {'x': val_data, 'y': val_labels})
    sio.savemat(current_dir + "/LSTMAE/LSTMAE_data_test.mat", {'x': test_data, 'y': test_labels})

    # 读取配置文件
    config_str = current_dir + "/LSTMAE/LSTMAE.yaml"
    config = rd.read_config(config_str)
    print("config: ")
    print(config)

    create_model(config)
    avg_train_loss, avg_val_loss = fit(train_iter, val_iter)  # 运行训练集和验证集
    errors, labels, preds, threshold = test(test_iter)

    print(f"Detection threshold={threshold:.4f}")  # 打印检测阈值
    # 计算预测精度 (preds 与真实标签 labels 的一致率)
    acc = (preds == labels).mean()
    print(f"Anomaly detection accuracy: {acc:.4f}")

    # 绘制重建误差分布
    plot_anomaly_distribution(errors, labels, threshold)
    print("LSTM-AE Finished.")

if __name__ == "__main__":
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    # 运行任务
    lstmae_task(device)