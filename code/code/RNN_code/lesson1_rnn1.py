# 导入若干工具包
import torch
import torch.nn as nn

# 实例化rnnr对象
# 第一个参数: input_size(输入张量x的维度)
# 第二个参数: hidden_size(隐藏层的维度, 隐藏层神经元数量)
# 第三个参数: num_layers(隐藏层的层数)
rnn = nn.RNN(5, 6, 1)

# 设定输入的张量x
# 第一个参数: sequence_length(输入序列的长度)
# 第二个参数: batch_size(批次的样本数)
# 第三个参数: input_size(输入张量x的维度)
input1 = torch.randn(1, 3, 5)

# 设定初始化的h0
# 第一个参数: num_layers * num_directions(层数*网络方向数)
# 第二个参数: batch_size(批次的样本数)
# 第三个参数: hidden_size(隐藏层的维度)
h0 = torch.randn(1, 3, 6)

# 输入张量放入RNN中, 得到输出结果
output, hn = rnn(input1, h0)

# print(output)
# print(output.shape)
# print(hn)
# print(hn.shape)

# ------------------------------------------------

# 实例化LSTM对象
# 第一个参数: input_size(输入张量x的维度)
# 第二个参数: hidden_size(隐藏层的维度, 隐藏层的神经元数量)
# 第三个参数: num_layers(隐藏层的层数)
lstm = nn.LSTM(5, 6, 2)

# 初始化输入张量x
# 第一个参数: sequence_length(输入序列的长度)
# 第二个参数: batch_size(批次的样本数量)
# 第三个参数: input_size(输入张量x的维度)
input1 = torch.randn(1, 3, 5)

# 初始化隐藏层张量h0, 和细胞状态c0
# 第一个参数: num_layers * num_directions(隐藏层的层数*方向数)
# 第二个参数: batch_size(批次的样本数量)
# 第三个参数: hidden_size(隐藏层的维度)
h0 = torch.randn(2, 3, 6)
c0 = torch.randn(2, 3, 6)

# 将input1, h0, c0输入lstm中, 得到输出张量结果
# output, (hn, cn) = lstm(input1, (h0, c0))
# print(output)
# print(output.shape)
# print(hn)
# print(hn.shape)
# print(cn)
# print(cn.shape)

# -------------------------------------------------

# 实例化GRU对象
# 第一个参数: input_size(输入张量x的维度)
# 第二个参数: hidden_size(隐藏层的维度, 隐藏层神经元的数量)
# 第三个参数: num_layers(隐藏层的层数)
gru = nn.GRU(5, 6, 2)

# 初始化输入张量input1
# 第一个参数: sequence_length(序列的长度)
# 第二个参数: batch_size(批次的样本个数)
# 第三个参数: input_size(输入张量x的维度)
input1 = torch.randn(1, 3, 5)

# 初始化隐藏层的张量h0
# 第一个参数: num_layers * num_directions(隐藏层的层数*方向数)
# 第二个参数: batch_size(批次的样本个数)
# 第三个参数: hidden_size(隐藏层的维度)
h0 = torch.randn(2, 3, 6)

# 将input1, h0输入GRU中, 得到输出张量结果
output, hn = gru(input1, h0)
print(output)
print(output.shape)
print(hn)
print(hn.shape)



















