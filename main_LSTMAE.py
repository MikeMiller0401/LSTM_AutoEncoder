from LSTMAE.Lib_LSTMAE import *
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
import readData as rd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def lstmae_task():
    # 设置随机种子，保证结果可复现
    # torch.manual_seed(42)
    # np.random.seed(42)

    anomaly_ratio = 0.05  # 异常比例
    num_of_sequences = 10000  # 数据长度
    X, Y = generate_data(num_of_sequences=num_of_sequences, sequence_len=50, anomaly_ratio=anomaly_ratio)

    # 分割为训练集
    X_train = X[:int(num_of_sequences * 0.6)]
    Y_train = Y[:int(num_of_sequences * 0.6)]

    # 分割为测试集
    X_val = X[int(num_of_sequences * 0.6):int(num_of_sequences * 0.8)]
    Y_val = Y[int(num_of_sequences * 0.6):int(num_of_sequences * 0.8)]

    # 分割为训练集
    X_test = X[int(num_of_sequences * 0.8):]
    Y_test = Y[int(num_of_sequences * 0.8):]


    ### 添加保存和读取 ###
    current_dir = Path.cwd()
    sio.savemat(current_dir.joinpath("LSTMAE/LSTMAE_data_raw.mat"), {'x': X, 'y': Y})
    sio.savemat(current_dir.joinpath("LSTMAE/LSTMAE_data.mat"), {'x': X_train, 'y': Y_train})
    sio.savemat(current_dir.joinpath("LSTMAE/LSTMAE_data_val.mat"), {'x': X_val, 'y': Y_val})
    sio.savemat(current_dir.joinpath("LSTMAE/LSTMAE_data_test.mat"), {'x': X_test, 'y': Y_test})

    data_str = current_dir.joinpath("LSTMAE/LSTMAE_data.mat")
    xy_data = rd.read_data(data_str)
    print("xy_data")

    data_str = current_dir.joinpath("LSTMAE/LSTMAE_data_val.mat")
    xy_vali_data = rd.read_data(data_str)
    print("xy_data")

    data_str = current_dir.joinpath("LSTMAE/LSTMAE_data_test.mat")
    xy_test_data = rd.read_data(data_str)
    print("xy_test_data")

    config_str = current_dir.joinpath("LSTMAE/LSTMAE.yaml")
    config = rd.read_config(config_str)
    print("config")
    print(config)

    print("create_model")
    create_model(config)

    xy_data_list = [xy_data['x'], xy_data['y']]
    xy_vali_data_list = [xy_vali_data['x'], xy_vali_data['y']]
    losses = fit(xy_data_list, xy_vali_data_list)

    # 添加模型保存
    save(current_dir.joinpath("LSTMAE/LSTMAE_model.pth"))

    # 绘制Loss
    train_losses = losses[0]
    val_losses = losses[1]
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.title("Train Loss")
    plt.plot(train_losses, label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Val Loss")
    plt.plot(val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    vali_rec_loss = predict_rec_loss(xy_vali_data['x'], 32)  # 验证集重建误差
    test_rec_loss = predict_rec_loss(xy_test_data['x'], 32)  # 测试集重建误差

    # 计算阈值和结果
    threshold = get_threshold(vali_rec_loss)
    print(f"最佳阈值: {threshold:.6f}")
    preds = (test_rec_loss > threshold).astype(int)
    # 计算指标
    accuracy = accuracy_score(Y_test, preds)
    precision = precision_score(Y_test, preds, zero_division=0)
    recall = recall_score(Y_test, preds, zero_division=0)
    f1 = f1_score(Y_test, preds, zero_division=0)
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 分数: {f1:.4f}")

    #  绘制检测图
    plt.figure(figsize=(12, 5))
    plt.plot(test_rec_loss, label="Reconstruction Error")
    plt.axhline(threshold, color='r', linestyle='--', label=f"Threshold={threshold:.4f}")
    anomaly_idx = np.where(Y_test == 1)[0]  # 标记真实异常点
    plt.scatter(anomaly_idx, test_rec_loss[anomaly_idx], color='g', marker='o', label="True Anomalies")
    pred_anomaly_idx = np.where(preds == 1)[0]  # 标记预测异常点
    plt.scatter(pred_anomaly_idx, test_rec_loss[pred_anomaly_idx], color='r', marker='x', label="Predicted Anomalies")
    plt.title("Anomaly Detection Visualization")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # 运行任务
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    lstmae_task()