import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from jinja2.optimizer import optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_data_with_anomalies(num_of_sequences=10000, sequence_len=50, anomaly_ratio=0.05):
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

def create_dataloaders_with_anomalies(batch_size, train_ratio=0.6, val_ratio=0.2):
    data, labels = create_data_with_anomalies()
    length = data.shape[0]

    # 按比例划分
    train_data = data[:int(length * train_ratio)]
    train_labels = labels[:int(length * train_ratio)]

    val_data = data[int(length * train_ratio):int(length * (train_ratio + val_ratio))]
    val_labels = labels[int(length * train_ratio):int(length * (train_ratio + val_ratio))]

    test_data = data[int(length * (train_ratio + val_ratio)):]
    test_labels = labels[int(length * (train_ratio + val_ratio)):]

    print(f"Datasets shapes: Train={train_data.shape}; Validation={val_data.shape}; Test={test_data.shape}")

    train_iter = torch.utils.data.DataLoader(LstmaeDataset(train_data, train_labels), batch_size=batch_size,shuffle=True)
    val_iter = torch.utils.data.DataLoader(LstmaeDataset(val_data, val_labels), batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(LstmaeDataset(test_data, test_labels), batch_size=batch_size,shuffle=False)

    return train_iter, val_iter, test_iter, train_data, train_labels, val_data, val_labels, test_data, test_labels

class LstmaeDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, seq_len):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len

        self.lstm_enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)

    def forward(self, x):
        out, (last_h_state, last_c_state) = self.lstm_enc(x)
        x_enc = last_h_state.squeeze(dim=0)
        x_enc = x_enc.unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_enc, out

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, seq_len, use_act):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.seq_len = seq_len
        self.use_act = use_act  # Parameter to control the last sigmoid activation - depends on the normalization used.
        self.act = nn.Sigmoid()

        self.lstm_dec = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, z):
        # z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, (hidden_state, cell_state) = self.lstm_dec(z)
        dec_out = self.fc(dec_out)
        if self.use_act:
            dec_out = self.act(dec_out)

        return dec_out, hidden_state

class LSTMAE(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_ratio, seq_len, use_act=True):
        super(LSTMAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.seq_len = seq_len

        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout_ratio, seq_len=seq_len)
        self.decoder = Decoder(input_size=input_size, hidden_size=hidden_size, dropout=dropout_ratio, seq_len=seq_len, use_act=use_act)

    def forward(self, x, return_last_h=False, return_enc_out=False):
        x_enc, enc_out = self.encoder(x)
        x_dec, last_h = self.decoder(x_enc)

        if return_last_h:
            return x_dec, last_h
        elif return_enc_out:
            return x_dec, enc_out
        return x_dec

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

        self.device = torch.device(config_device if torch.cuda.is_available() else 'cpu')

        self.model = LSTMAE(input_size=config_input_size,
                            hidden_size=config_hidden_size,
                            dropout_ratio=config_dropout_ratio,
                            seq_len=config_seq_len).to(self.device)

        self.optimizer = getattr(torch.optim, 'Adam')(params=self.model.parameters(), lr=0.001, weight_decay=0)
        self.criterion = nn.MSELoss(reduction='sum')

    # 单个 batch 的训练
    def train_with_batch(self, train_batch_data, device):
        model = self.model.train()
        criterion = self.criterion
        optimizer = self.optimizer


        # 解包 batch 数据
        if isinstance(train_batch_data, (list, tuple)) and len(train_batch_data) == 2:
            batch_x, batch_y = train_batch_data
        else:
            batch_x = train_batch_data

        if isinstance(batch_x, torch.Tensor):
            tensor_x = batch_x.float().to(device)
        else:
            tensor_x = torch.tensor(batch_x, dtype=torch.float, device=device)

        # 前向传播
        outputs = model(tensor_x)
        loss = criterion(outputs, tensor_x)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item() * tensor_x.size(0)  # 返回总损失（方便累积后再求平均）

    def train(self, train_iter, val_iter):
        num_epochs = 100 # 训练总epoch数量
        log_interval = 100 # 记录的log间隔次数
        for epoch in range(1, num_epochs + 1):
            # ====== Training ======
            self.model.train()
            total_loss = 0
            num_samples = 0

            for batch_idx, batch_data in enumerate(train_iter, 1):
                batch_loss = self.train_with_batch(batch_data, self.device)
                total_loss += batch_loss
                if isinstance(batch_data, (list, tuple)):
                    batch_size = len(batch_data[0])
                else:
                    batch_size = len(batch_data)
                num_samples += batch_size

                if batch_idx % log_interval == 0:
                    print(f"Train Epoch: {epoch} [{num_samples}/{len(train_iter.dataset)} "
                          f"({100. * num_samples / len(train_iter.dataset):.0f}%)]\t"
                          f"Loss: {total_loss / num_samples:.6f}")

            avg_train_loss = total_loss / len(train_iter.dataset)
            print(f"Epoch {epoch} - Train Average Loss: {avg_train_loss:.6f}")

            # ====== Validation ======
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_data in val_iter:
                    if isinstance(batch_data, (list, tuple)):
                        data = batch_data[0]
                    else:
                        data = batch_data
                    data = torch.tensor(data, dtype=torch.float).to(self.device)

                    outputs = self.model(data)
                    loss = self.criterion(outputs, data)
                    val_loss += loss.item() * data.size(0)

            avg_val_loss = val_loss / len(val_iter.dataset)
            print(f"Epoch {epoch} - Validation Average Loss: {avg_val_loss:.6f}")

        return avg_train_loss, avg_val_loss

    def eval_model(criterion, model, model_type, val_iter, mode='Validation'):
        model.eval()
        loss_sum = 0
        correct_sum = 0
        with torch.no_grad():
            for data in val_iter:
                if len(data) == 2:
                    data, labels = data[0].to(device), data[1].to(device)
                else:
                    data = data.to(device)

                model_out = model(data)
                if model_type == 'LSTMAE_CLF':
                    model_out, out_labels = model_out
                    pred = out_labels.max(1, keepdim=True)[1]
                    correct_sum += pred.eq(labels.view_as(pred)).sum().item()
                    # Calculate loss
                    mse_loss, ce_loss = criterion(model_out, data, out_labels, labels)
                    loss = mse_loss + ce_loss
                elif model_type == 'LSTMAE_PRED':
                    # For S&P prediction
                    model_out, preds = model_out
                    labels = data.squeeze()[:, 1:]  # Take x_t+1 as y_t
                    preds = preds[:, :-1]  # Take preds up to T-1
                    mse_rec, mse_pred = criterion(model_out, data, preds, labels)
                    loss = mse_rec + mse_pred
                else:
                    # Calculate loss for none clf models
                    loss = criterion(model_out, data)

                loss_sum += loss.item()
        val_loss = loss_sum / len(val_iter.dataset)
        val_acc = round(correct_sum / len(val_iter.dataset) * 100, 2)
        acc_out_str = f'; Average Accuracy: {val_acc}' if model_type == 'LSTMAECLF' else ''
        print(f' {mode}: Average Loss: {val_loss}{acc_out_str}')
        return val_loss, val_acc



class Lib_LSTMAE:
    model = None

def create_model(config):
    if config is not None:
        Lib_LSTMAE.model = LSTMAE_Method(config)
    else:
        print("Config is None")

def train_model(criterion, epoch, model, model_type, optimizer, train_iter, batch_size, clip_val, log_interval, scheduler=None):
    """
    Function to run training epoch
    :param criterion: loss function to use
    :param epoch: current epoch index
    :param model: pytorch model object
    :param model_type: model type (only ae/ ae+clf), used to know if needs to calculate accuracy
    :param optimizer: optimizer to use
    :param train_iter: train dataloader
    :param batch_size: size of batch (for logging)
    :param clip_val: gradient clipping value
    :param log_interval: interval to log progress
    :param scheduler: learning rate scheduler, optional.
    :return mean train loss (and accuracy if in clf mode)
    """
    model.train()
    loss_sum = 0
    pred_loss_sum = 0
    correct_sum = 0

    num_samples_iter = 0
    for batch_idx, data in enumerate(train_iter, 1):
        if len(data) == 2:
            data, labels = data[0].to(device), data[1].to(device)
        else:
            data = data.to(device)
        # Zero gradients
        optimizer.zero_grad()

        num_samples_iter += len(data)  # Count number of samples seen in epoch (used for later statistics)

        # Forward pass & loss calculation
        model_out = model(data)
        if model_type == 'LSTMAE_CLF':
            # For MNIST classifier
            model_out, out_labels = model_out
            pred = out_labels.max(1, keepdim=True)[1]
            correct_sum += pred.eq(labels.view_as(pred)).sum().item()
            # Calculate loss
            mse_loss, ce_loss = criterion(model_out, data, out_labels, labels)
            loss = mse_loss + ce_loss
        elif model_type == 'LSTMAE_PRED':
            # For S&P prediction
            model_out, preds = model_out
            labels = data.squeeze()[:, 1:]  # Take x_t+1 as y_t
            preds = preds[:, :-1]  # Take preds up to T-1
            mse_rec, mse_pred = criterion(model_out, data, preds, labels)
            loss = mse_rec + mse_pred
            pred_loss_sum += mse_pred.item()
        else:
            # Calculate loss
            loss = criterion(model_out, data)

        # Backward pass
        loss.backward()
        loss_sum += loss.item()

        # Gradient clipping
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

        # Update model params
        optimizer.step()

        # LR scheduler step
        if scheduler is not None:
            scheduler.step()

        # print progress
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, num_samples_iter, len(train_iter.dataset),
                100. * num_samples_iter / len(train_iter.dataset), loss_sum / num_samples_iter))
    train_loss = loss_sum / len(train_iter.dataset)
    train_pred_loss = pred_loss_sum / len(train_iter.dataset)
    train_acc = round(correct_sum / len(train_iter.dataset) * 100, 2)
    acc_out_str = f'; Average Accuracy: {train_acc}' if model_type == 'LSTMAECLF' else ''
    print(f'Train Average Loss: {train_loss}{acc_out_str}')

    return train_loss, train_acc, train_pred_loss

def eval_model(criterion, model, model_type, val_iter, mode='Validation'):
    """
    Function to run validation on given model
    :param criterion: loss function
    :param model: pytorch model object
    :param model_type: model type (only ae/ ae+clf), used to know if needs to calculate accuracy
    :param val_iter: validation dataloader
    :param mode: mode: 'Validation' or 'Test' - depends on the dataloader given.Used for logging
    :return mean validation loss (and accuracy if in clf mode)
    """
    # Validation loop
    model.eval()
    loss_sum = 0
    correct_sum = 0
    with torch.no_grad():
        for data in val_iter:
            if len(data) == 2:
                data, labels = data[0].to(device), data[1].to(device)
            else:
                data = data.to(device)

            model_out = model(data)
            if model_type == 'LSTMAE_CLF':
                model_out, out_labels = model_out
                pred = out_labels.max(1, keepdim=True)[1]
                correct_sum += pred.eq(labels.view_as(pred)).sum().item()
                # Calculate loss
                mse_loss, ce_loss = criterion(model_out, data, out_labels, labels)
                loss = mse_loss + ce_loss
            elif model_type == 'LSTMAE_PRED':
                # For S&P prediction
                model_out, preds = model_out
                labels = data.squeeze()[:, 1:]  # Take x_t+1 as y_t
                preds = preds[:, :-1]  # Take preds up to T-1
                mse_rec, mse_pred = criterion(model_out, data, preds, labels)
                loss = mse_rec + mse_pred
            else:
                # Calculate loss for none clf models
                loss = criterion(model_out, data)

            loss_sum += loss.item()
    val_loss = loss_sum / len(val_iter.dataset)
    val_acc = round(correct_sum / len(val_iter.dataset) * 100, 2)
    acc_out_str = f'; Average Accuracy: {val_acc}' if model_type == 'LSTMAECLF' else ''
    print(f' {mode}: Average Loss: {val_loss}{acc_out_str}')
    return val_loss, val_acc

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

def fit(train_iter, val_iter):
    print("fit with test model")
    if train_iter is not None:
        print("Train")
        avg_train_loss, avg_val_loss = Lib_LSTMAE.model.train(train_iter, val_iter)
    elif train_iter is None:
        print("eval")
        losses = Lib_LSTMAE.model.train_with_eval()
    else:
        print("Error")
    return avg_train_loss, avg_val_loss