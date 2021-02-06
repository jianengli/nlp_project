import torch
import torch.nn as nn
import torch.nn.functional as F


# 参数一: 输入张量的词嵌入维度5, 参数二: 隐藏层的维度(也就是神经元的个数), 参数三: 网络层数
rnn = nn.RNN(5, 6, 2)

# 参数一: sequence_length序列长度, 参数二: batch_size样本个数, 参数三: 词嵌入的维度, 和RNN第一个参数匹配
input1 = torch.randn(1, 3, 5)

# 参数一: 网络层数, 和RNN第三个参数匹配, 参数二: batch_size样本个数, 参数三: 隐藏层的维度, 和RNN第二个参数匹配
h0 = torch.randn(2, 3, 6)


output, hn = rnn(input1, h0)
# print(output.shape)
# torch.Size([1, 3, 6])
# print(hn.shape)
# torch.Size([2, 3, 6])

# -------------------------------------

rnn1 = nn.LSTM(5, 6, 2)
input1 = torch.randn(1, 3, 5)
h0 = torch.randn(2, 3, 6)
c0 = torch.randn(2, 3, 6)

output, (hn, cn) = rnn1(input1, (h0, c0))

# print(output)
# print(output.shape)
# print(hn)
# print(hn.shape)
# print(cn)
# print(cn.shape)

# -------------------------------------

class Attention(nn.Module):
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super(Attention, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1 = value_size1
        self.value_size2 = value_size2
        self.output_size = output_size
        
        self.attn = nn.Linear(self.query_size + self.key_size, self.value_size1)
        
        self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)
        
    def forward(self, Q, K, V):
        attn_weights = F.softmax(self.attn(torch.cat((Q[0], K[0]), 1)), dim=1)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), V)
        
        output = torch.cat((Q[0], attn_applied[0]), 1)
        
        output1 = self.attn_combine(output).unsqueeze(0)
        
        return output1, attn_weights


query_size = 32
key_size = 32
value_size1 = 32
value_size2 = 64
output_size = 64
attn = Attention(query_size, key_size, value_size1, value_size2, output_size)
Q = torch.randn(1, 1, 32)
K = torch.randn(1, 1, 32)
V = torch.randn(1, 32, 64)
res = attn(Q, K, V)
# print(res[0])
# print(res[0].shape)
# print('*****')
# print(res[1])
# print(res[1].shape)

# --------------------------------------------