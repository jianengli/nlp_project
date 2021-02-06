import torch
import torch.nn as nn

# 实例化RNN对象
# 第一个参数: input_size (输入张量x的维度)
# 第二个参数: hidden_size (隐藏层的维度, 隐藏层的神经元的数量)
# 第三个参数: num_layers (隐藏层的层数)
# rnn = nn.RNN(5, 6, 1)

# 初始化输入张量input
# 第一个参数: sequence_length (输入序列的长度)
# 第二个参数: batch_size (批次的样本数)
# 第三个参数: input_size (输入张量的维度)
# input = torch.randn(1, 3, 5)

# 初始化h0
# 第一个参数: num_layers * num_directions (隐藏层的层数 * 方向数)
# 第二个参数: batch_size (批次的样本数)
# 第三个参数: hidden_size (隐藏层的维度)
# h0 = torch.randn(1, 3, 6)

# RNN每次接收两个输入张量, 得到两个输出张量
# output, hn = rnn(input, h0)

# 打印结果张量
# print(output)
# print(output.shape)
# print(hn)
# print(hn.shape)


# 实例化GRU对象
# 第一个参数: input_size (输入张量的维度)
# 第二个参数: hidden_size (隐藏层的维度, 隐藏层神经元的个数)
# 第三个参数: num_layers (隐藏层的层数)
gru = nn.GRU(5, 6, 2)

# 初始化输入张量input
# 第一个参数: sequence_length (输入序列的长度)
# 第二个参数: batch_size (批次的样本数量)
# 第三个参数: input_size (输入张量的维度)
input = torch.randn(1, 3, 5)

# 初始化隐藏层的张量h0
# 第一个参数: num_layers * num_directions (隐藏层的层数 * 方向数)
# 第二个参数: batch_size (批次的样本数量)
# 第三个参数: hidden_size (隐藏层的维度)
h0 = torch.randn(2, 3, 6)

# GRU接收两个输入张量, 得到两个输出张量
output, hn = gru(input, h0)

# 打印输出的张量和形状
print(output)
print(output.shape)
print(hn)
print(hn.shape)



















