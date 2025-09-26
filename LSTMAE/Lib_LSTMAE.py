import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import readData as rd
import seaborn as sns


def generate_data(num_of_sequences=10000, sequence_len=50, anomaly_ratio=0.05):
    # 正常数据（均匀分布）
    X = torch.rand((num_of_sequences, sequence_len, 1))
    y = torch.zeros(num_of_sequences)  # 默认全是正常

    # 随机挑一些样本注入异常
    num_anomalies = int(num_of_sequences * anomaly_ratio)
    anomaly_indices = np.random.choice(num_of_sequences, num_anomalies, replace=False)

    for idx in anomaly_indices:
        seq = X[idx].clone()

        # 异常类型 1: 突变 spike
        if np.random.rand() > 0.5:
            pos = np.random.randint(0, sequence_len)
            seq[pos:] += 2.0
        # 异常类型 2: 整体偏移 shift
        else:
            seq += 1.5

        X[idx] = seq
        y[idx] = 1  # 标记异常

    # 转换为 numpy，和第一个函数保持一致
    X = X.numpy()
    y = y.numpy()

    return X, y

def get_threshold(errors):
    threshold = errors.mean() + 3 * errors.std()
    return threshold

class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_ratio, seq_len, use_act=True):
        super(LSTMAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.seq_len = seq_len
        self.use_act = use_act

        # ===== Encoder =====
        self.lstm_enc = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout_ratio,
            batch_first=True
        )

        # ===== Decoder =====
        self.lstm_dec = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout_ratio,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, input_size)
        self.act = nn.Sigmoid()

    def forward(self, x, return_last_h=False, return_enc_out=False):
        # ---------- Encoder ----------
        enc_out, (last_h_state, last_c_state) = self.lstm_enc(x)
        # 使用最后一个 hidden state，重复到序列长度
        x_enc = last_h_state.squeeze(0)                 # (batch, hidden)
        x_enc = x_enc.unsqueeze(1).repeat(1, x.shape[1], 1)  # (batch, seq_len, hidden)

        # ---------- Decoder ----------
        dec_out, (hidden_state, cell_state) = self.lstm_dec(x_enc)
        dec_out = self.fc(dec_out)  # (batch, seq_len, input_size)

        if self.use_act:
            dec_out = self.act(dec_out)

        # ---------- 返回控制 ----------
        if return_last_h:
            return dec_out, hidden_state
        elif return_enc_out:
            return dec_out, enc_out
        return dec_out

    def get_reconstruction_error(self, x):
        """计算输入数据的重构误差"""
        with torch.no_grad():
            x_recon = self.forward(x)
            # 使用MSE作为重构误差
            error = torch.mean(torch.square(x - x_recon), dim=1)
        return error

class LSTMAE_Method():
    def __init__(self, config):
        self.config = config

        if 'input_size' in config:
            config_input_size = config.input_size
        else:
            print("no input_size value")

        if 'hidden_size' in config:
            config_hidden_size = config.hidden_size
        else:
            print("no hidden_size value")

        if 'dropout_ratio' in config:
            config_dropout_ratio = config.dropout_ratio
        else:
            print("no dropout_ratio value")

        if 'seq_len' in config:
            config_seq_len = config.seq_len
        else:
            print("no seq_len value")

        if 'device' in config:
            config_device = config.device
        else:
            print("no num_layers value")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
        print(self.device)

        self.model = LSTMAE(input_size=config_input_size,
                            hidden_size=config_hidden_size,
                            dropout_ratio=config_dropout_ratio,
                            seq_len=config_seq_len).to(self.device)

        self.optimizer = getattr(torch.optim, 'Adam')(params=self.model.parameters(), lr=0.001, weight_decay=0)
        self.criterion = nn.MSELoss(reduction='sum')

    # 单个 batch 的训练
    def train_with_batch(self, train_batch_data):
        self.model.train()
        train_x = train_batch_data[0]
        tensor_x = torch.tensor(train_x, dtype=torch.float).to(self.device)
        outputs = self.model(tensor_x)
        loss = self.criterion(outputs, tensor_x)
        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数
        x = tensor_x.size(0)
        y = loss.item()
        return loss.item() * tensor_x.size(0)

    def train(self, train_data):
        epochs = 20
        batch_size = 32
        train_losses = []
        X_train = train_data[0]

        # 批量训练
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                train_batch_loss = self.train_with_batch([batch_X])
                train_loss += train_batch_loss
            # 计算平均训练损失
            train_loss /= len(X_train)
            train_losses.append(train_loss)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}')
        return train_losses

    def train_with_test(self, train_data, test_data):
        epochs = 100
        batch_size = 32
        train_losses = []
        test_losses = []
        X_train = train_data[0]
        X_test = test_data[0]

        # 批量训练
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                train_batch_loss = self.train_with_batch([batch_X])
                train_loss += train_batch_loss
            # 计算平均训练损失
            train_loss /= len(X_train)
            train_losses.append(train_loss)
            # 测试模式
            test_preds, test_loss = self.eval_with_test(X_test)
            test_loss /= len(X_test)
            test_losses.append(test_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
        return train_losses, test_losses

    def eval_with_test(self, x_test_data):
        self.model.eval()
        tensor_test_x = torch.tensor(x_test_data, dtype=torch.float).to(self.device)
        with torch.no_grad():
            test_outputs = self.model(tensor_test_x)
            test_loss = self.criterion(test_outputs, tensor_test_x).item()
            test_preds = test_outputs.cpu().numpy()
        return test_preds, test_loss

    def eval_rec_loss(self, x_test_data, batch_size):
        self.model.eval()
        tensor_test_x = torch.tensor(x_test_data, dtype=torch.float).to(self.device)

        rec_errors = []
        with torch.no_grad():
            for i in range(0, len(x_test_data), batch_size):
                batch_X = tensor_test_x[i:i + batch_size]
                batch_errors = self.model.get_reconstruction_error(batch_X)
                rec_errors.extend(batch_errors.cpu().numpy())
        rec_errors = np.array(rec_errors)/len(x_test_data)

        return rec_errors


    def test(self, test_iter):
        self.model.eval()
        errors, labels = [], []
        with torch.no_grad():
            for seq, lbl in test_iter:
                seq = seq.to(self.device)
                rec = self.model(seq)

                # 对每个样本单独计算重建误差 (MSE)
                batch_errors = torch.mean((rec - seq) ** 2, dim=(1, 2))  # shape: [batch_size]
                errors.extend(batch_errors.cpu().numpy())
                labels.extend(lbl.cpu().numpy())

        errors = np.array(errors)
        labels = np.array(labels)
        threshold = errors.mean() + 3 * errors.std()
        preds = (errors > threshold).astype(int)  # 预测结果：1=异常
        return errors, labels, preds, threshold

class Lib_LSTMAE:
    model = None

def print_model():
    print(type(Lib_LSTMAE.model))
    if Lib_LSTMAE.model is not None:
        print(Lib_LSTMAE.model.model)

def create_model(config):
    if config is not None:
        Lib_LSTMAE.model = LSTMAE_Method(config)
    else:
        print("Config is None")

def fit(X_train, X_test=None):
    print("fit with test model")
    if X_test is None:
        if X_train is not None:
            losses = Lib_LSTMAE.model.train(X_train)
        else:
            print("X_train is None")

    else:
        if X_train is not None:
            losses = Lib_LSTMAE.model.train_with_test(X_train, X_test)
        else:
            print("X_train is None")
    return losses

def predict_rec_loss(X_test, batch_size):
    print("predict model")
    if X_test is not None:
        rec_errors = Lib_LSTMAE.model.eval_rec_loss(X_test, batch_size)
        return rec_errors

    else:
        print("X_test is None")
        return None

def save(file_name):
    torch.save(Lib_LSTMAE.model, file_name)

def load(file_name):
    Lib_LSTMAE.model = torch.load(file_name, weights_only=False)

if __name__ == "__main__":

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

    ### 添加保存和读取 ###


    config_str = "E:\PycharmProjects\LSTM_AutoEncoder\LSTMAE\LSTMAE.yaml"
    config = rd.read_config(config_str)
    print("config")
    print(config)

    print_model()
    print("create_model")
    create_model(config)
    print_model()

    xy_data_list = [X_train, Y_train]
    xy_vali_data_list = [X_val, Y_val]
    losses = fit(xy_data_list, xy_vali_data_list)

    ### 添加模型保存 ###

    ### 添加模型保存 ###

    train_losses = losses[0]
    val_losses = losses[1]

    plt.figure(figsize=(12, 6))
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

    vali_rec_loss = predict_rec_loss(X_val, 32)
    test_rec_loss = predict_rec_loss(X_test, 32)

    threshold = get_threshold(vali_rec_loss)
    print(f"最佳阈值: {threshold:.6f}")
    preds = (test_rec_loss > threshold).astype(int)

    # 绘制重构误差分布
    plt.figure(figsize=(10, 6))
    sns.histplot(test_rec_loss[Y_test == 0], kde=True, bins=50, label='正常样本', color='green', alpha=0.6)
    sns.histplot(test_rec_loss[Y_test == 1], kde=True, bins=50, label='异常样本', color='red', alpha=0.6)
    plt.axvline(x=threshold, color='blue', linestyle='--', label=f'最佳阈值: {threshold:.4f}')
    plt.xlabel('重构误差 (MSE)')
    plt.ylabel('频率')
    plt.title('正常与异常样本的重构误差分布')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

