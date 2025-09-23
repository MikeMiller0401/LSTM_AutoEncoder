from LSTMAE.Lib_LSTMAE import *
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.io as sio
import readData as rd
def lstmae_task(device):
    print("LSTM-AE Begin.")
    current_dir = str(Path.cwd())

    # 建立&保存 数据集
    train_iter, val_iter, test_iter, train_data, train_labels, val_data, val_labels, test_data, test_labels = create_dataloaders_with_anomalies(batch_size=128)
    sio.savemat(current_dir + "\LSTMAE\LSTMAE_data_train.mat", {'x': train_data, 'y': train_labels})
    sio.savemat(current_dir + "\LSTMAE\LSTMAE_data_val.mat", {'x': val_data, 'y': val_labels})
    sio.savemat(current_dir + "\LSTMAE\LSTMAE_data_test.mat", {'x': test_data, 'y': test_labels})

    # 读取配置文件
    config_str = current_dir + "\LSTMAE\LSTMAE.yaml"
    config = rd.read_config(config_str)
    print("config: ")
    print(config)

    create_model(config)
    losses = fit(train_iter, val_iter)



    # 实例化模型
    print("create_model: ")
    model = LSTMAE(input_size=config.input_size, hidden_size=config.hidden_size,
                   dropout_ratio=config.dropout_ratio, seq_len=config.seq_len).to(device=config.device)
    optimizer = getattr(torch.optim, 'Adam')(params=model.parameters(), lr=0.001, weight_decay=0)
    criterion = torch.nn.MSELoss(reduction='sum')

    for epoch in range(100):
        train_model(criterion, epoch, model, 'LSTMAE', optimizer, train_iter, 128, None, 10)
        eval_model(criterion, model, 'LSTMAE', val_iter)

    errors, labels, preds, threshold = detect_anomalies(model, test_iter, criterion)

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