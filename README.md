# 请注意，修订仅存在于python lstm_ae_toy.py文件

## 修改的内容
1. 加入了异常点用于Demo
2. 修改了输入数据的数量
3. 增加了图片输出

## 补充运行细节
1. 需要安装Pytorch，请根据CUDA版本安装
2. 其他需要安装的包:
```commandline
pip install numpy matplotlib pandas scikit-learn
```
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
