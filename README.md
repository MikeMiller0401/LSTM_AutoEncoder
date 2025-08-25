# LSTM AutoEncoder Toy Example 技术文档
请注意，修订仅存在于python lstm_ae_toy.py文件
1. 加入了异常点用于Demo
2. 修改了输入数据的数量
3. 增加了图片输出

补充运行细节

1. 需要安装Pytorch，请根据CUDA版本安装
2. 其他需要安装的包:
```commandline
pip install numpy matplotlib pandas scikit-learn
```
## 本项目实现了一个基于 **LSTM 自编码器 (LSTM-AE)** 的 **时间序列异常检测示例**。  

主要功能包括：  
- 数据集构建（正常序列 & 异常序列）  
- 模型训练与验证  
- 异常检测与可视化  
- 超参数网格搜索  

---

## 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--batch-size` | int | 128 | 每个 batch 的样本数量，影响训练效率和内存占用 |
| `--epochs` | int | 1000 | 训练总轮数，每一轮完整遍历训练集一次 |
| `--optim` | str | Adam | 优化器类型（可选：Adam, SGD 等） |
| `--hidden-size` | int | 256 | LSTM 隐藏层维度，决定模型容量 |
| `--lr` | float | 0.001 | 学习率，控制参数更新步长 |
| `--input-size` | int | 1 | 输入特征维度（单变量时间序列=1，多变量可设更大） |
| `--dropout` | float | 0.0 | dropout 比例，用于防止过拟合 |
| `--wd` | float | 0 | 权重衰减系数（L2 正则化） |
| `--grad-clipping` | float | None | 梯度裁剪阈值，防止梯度爆炸 |
| `--log-interval` | int | 10 | 每多少个 batch 输出一次日志 |
| `--model-type` | str | LSTMAE | 模型类型，目前仅支持 `LSTMAE` |
| `--model-dir` | str | trained_models | 模型 checkpoint 保存目录 |
| `--seq-len` | int | 50 | 输入序列长度（时间步数量） |
| `--run-grid-search` | flag | False | 是否运行超参数网格搜索 |

---

## 数据相关函数

### `create_toy_data(num_of_sequences=10000, sequence_len=50)`
生成随机 toy 数据（正常序列）。  
- **参数**:  
  - `num_of_sequences` → 序列数量  
  - `sequence_len` → 每个序列的长度  
- **返回**: `toy_data (torch.Tensor)`  

---

### `create_toy_data_with_anomalies(num_of_sequences=10000, sequence_len=50, anomaly_ratio=0.05)`
生成带异常的 toy 数据集，包括 `spike`（突变）和 `shift`（整体偏移）。  
- **返回**: `(toy_data, labels)` 其中 `labels=0` 正常，`1` 异常  

---

### `toy_dataset(torch.utils.data.Dataset)`
自定义 PyTorch 数据集。  
- `__len__` → 返回样本数  
- `__getitem__(index)` → 返回 `(data, label)`  

---

### `create_dataloaders(batch_size, train_ratio=0.6, val_ratio=0.2)`
构建 **正常数据集** 的 train/val/test DataLoader。  

---

### `create_dataloaders_with_anomalies(batch_size, train_ratio=0.6, val_ratio=0.2)`
构建 **带异常数据集** 的 train/val/test DataLoader。  

---

## 模型训练与评估函数

### `train_model(...)` *(外部导入)*
- 执行一个 epoch 的训练  
- 支持梯度裁剪、日志打印  

### `eval_model(...)` *(外部导入)*
- 在验证集上评估模型性能  
- 返回平均验证损失和准确率  

### `hyper_params_grid_search(train_iter, val_iter, criterion)`
- 执行超参数网格搜索（lr × hidden_size × grad_clip）  
- 自动找到验证集损失最优的参数组合  

---

## 异常检测与可视化

### `detect_anomalies(model, test_iter, criterion, threshold=None)`
利用重建误差进行异常检测。  
- **返回**:  
  - `errors` → 重建误差  
  - `labels` → 真实标签  
  - `preds` → 预测标签（0=正常，1=异常）  
  - `threshold` → 使用的阈值  

---

### `plot_anomaly_distribution(errors, labels, threshold)`
绘制 **样本编号 vs 重建误差** 散点图。  

---

### `plot_toy_data(toy_example, description, color='b')`
绘制单条 toy 时间序列。  

---

### `plot_orig_vs_reconstructed(model, test_iter, num_to_plot=2)`
绘制 **原始 vs 重建** 的时间序列对比图。  

---

## 主程序入口

### `main()`
1. 生成带异常的 toy 数据集  
2. 构建模型 `LSTMAE`  
3. 设置优化器 & 损失函数  
4. 训练 + 验证  
5. 在测试集上执行异常检测  
6. 可视化检测结果


## 横线以下是原始README文件


----------------
# LSTM Auto-Encoder (LSTM-AE) implementation in Pytorch
The code implements three variants of LSTM-AE:
1. Regular LSTM-AE for reconstruction tasks (LSTMAE.py)
2. LSTM-AE + Classification layer after the decoder (LSTMAE_CLF.py)
3. LSTM-AE + prediction layer on top of the encoder (LSTMAE_PRED.py)

To test the implementation, we defined three different tasks:

Toy example (on random uniform data) for sequence reconstruction:
```
python lstm_ae_toy.py
```

MNIST reconstruction + classification:
```
python lstm_ae_mnist.py
```

SnP stock daily graph reconstruction + price prediction:
```
python lstm_ae_snp500.py
```
