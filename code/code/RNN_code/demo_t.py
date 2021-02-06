import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


# embedding = nn.Embedding(10, 3)
# input1 = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# print(embedding(input1))

# embedding = nn.Embedding(10, 3, padding_idx=0)
# input1 = torch.LongTensor([[0, 2, 3, 5]])
# print(embedding(input1))

# -------------------------------------

class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.lut = nn.Embedding(vocab, d_model)
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# x = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
d_model = 512
vocab = 1000
# emb = Embedding(d_model, vocab)
# res = emb(x)
# print(res)
# print(res.shape)

# -------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embedding_dim)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


# x = res
dropout = 0.1
max_len = 60
# pe = PositionalEncoding(d_model, dropout, max_len)
# pe_result = pe(x)
# print(pe_result)
# print(pe_result.shape)
        
# -------------------------------------------

# plt.figure(figsize=(15, 5))

# pe = PositionalEncoding(20, 0)
# y = pe(Variable(torch.zeros(1, 100, 20)))

# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d"%p for p in [4, 5, 6, 7]])

# -------------------------------------------

def subsequent_mask(size):
    attn_shape = (1, size, size)
    
    sub_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    
    return torch.from_numpy(1 - sub_mask)

sm = subsequent_mask(5)
# print(sm)

# plt.figure(figsize=(5, 5))
# plt.imshow(subsequent_mask(20)[0])

# -------------------------------------------

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    
    scores = torch.matmul(query, key.transpose(-2, -1)) // math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


# query = key = value = pe_result
# attn, p_attn = attention(query, key, value)
# print(attn)
# print(attn.shape)
# print('*****')
# print(p_attn)
# print(p_attn.shape)

# -------------------------------------------

def clone(model, N):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=dropout)
        self.d_k = embedding_dim // head
        
        self.linears = clone(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
    
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        
        batch_size = query.size(0)
        
        query, key, value = \
               [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) 
                for model, x in zip(self.linears, (query, key, value))]
        
        x, self.attn = attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        
        return self.linears[-1](x)


head = 8
embedding_dim = 512
dropout = 0.2

# query = key = value = pe_result
# mask = Variable(torch.zeros(8, 4, 4))

# mha = MultiHeadAttention(head, embedding_dim, dropout)
# mha_result = mha(query, key, value, mask)
# print(mha_result)
# print(mha_result.shape)     

# -----------------------------------------------

class PositionalwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# x = mha_result
d_model = 512
d_ff = 64
dropout = 0.2

# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
# ff_result = ff(x)
# print(ff_result)
# print(ff_result.shape)

# ------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


features = d_model = 512
eps = 1e-6
# x = ff_result
# ln = LayerNorm(features, eps)
# ln_result = ln(x)
# print(ln_result)
# print(ln_result.shape)

# ------------------------------------------------

class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SubLayerConnection, self).__init__()
        self.size = size
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(size)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


size = 512
dropout = 0.2
# x = pe_result
# self_attn = MultiHeadAttention(head, d_model)
# sublayer = lambda x: self_attn(x, x, x, mask)

# sc = SubLayerConnection(size, dropout)
# sc_result = sc(x, sublayer)
# print(sc_result)
# print(sc_result.shape)

# -----------------------------------------------

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = clone(SubLayerConnection(size, dropout), 2)
    
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

size = 512
head = 8
d_model = 512
d_ff = 64
# x = pe_result
dropout = 0.2
# self_attn = MultiHeadAttention(head, d_model)
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
# mask = Variable(torch.zeros(8, 4, 4))

# el = EncoderLayer(size, self_attn, ff, dropout)
# el_result = el(x, mask)
# print(el_result)
# print(el_result.shape)

# ----------------------------------------------

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

size = 512
c = copy.deepcopy
# attn = MultiHeadAttention(head, d_model)
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
# layer = EncoderLayer(size, c(attn), c(ff), dropout)

N = 8

# en = Encoder(layer, N)
# en_result = en(x, mask)
# print(en_result)
# print(en_result.shape)

# ----------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = clone(SubLayerConnection(size, dropout), 3)
    
    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        return self.sublayer[2](x, self.feed_forward)


head = 8
size = 512
d_model = 512
d_ff = 64
dropout = 0.2
# self_attn = MultiHeadAttention(head, d_model, dropout)
# src_attn = MultiHeadAttention(head, d_model, dropout)
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
# x = pe_result
# memory = en_result
# source_mask = target_mask = mask

# dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
# dl_result = dl(x, memory, source_mask, target_mask)
# print(dl_result)
# print(dl_result.shape)

# -------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


size = 512
d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
# attn = MultiHeadAttention(head, d_model, dropout)
# ff = PositionalwiseFeedForward(d_model, d_ff, dropout)

# layer = DecoderLayer(size, c(attn), c(attn), c(ff), dropout)
N = 8
# x = pe_result
# memory = en_result
# mask = Variable(torch.zeros(8, 4, 4))
# source_mask = target_mask = mask

# de = Decoder(layer, N)
# de_result = de(x, memory, source_mask, target_mask)
# print(de_result)
# print(de_result.shape)

# -----------------------------------------------------

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.project = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


d_model = 512
vocab_size = 1000

# x = de_result
# gen = Generator(d_model, vocab_size)
# gen_result = gen(x)
# print(gen_result)
# print(gen_result.shape)

# ------------------------------------------------------

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator
    
    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)
    
    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, 
                            target_mask)
    
    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask, 
                           target, target_mask)


# vocab_size = 1000
# d_model = 512
# encoder = en
# decoder = de
# source_embed = nn.Embedding(vocab_size, d_model)
# target_embed = nn.Embedding(vocab_size, d_model)
# generator = gen
# source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
# source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

# ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
# ed_result = ed(source, target, source_mask, target_mask)
# print(ed_result)
# print(ed_result.shape)

# --------------------------------------------------------

def make_model(source_vocab, target_vocab, N=6, d_model=512,
               d_ff=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    
    attn = MultiHeadAttention(head, d_model, dropout)
    
    ff = PositionalwiseFeedForward(d_model, d_ff, dropout)
    
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embedding(d_model, source_vocab), c(position)),
            nn.Sequential(Embedding(d_model, target_vocab), c(position)),
            Generator(d_model, target_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
            
    return model


source_vocab = 11
target_vocab = 11
N = 6

# if __name__ == '__main__':
#     res = make_model(source_vocab, target_vocab, N)
#     print(res)

# ----------------------------------------------------------

from pyitcast.transformer_utils import Batch
from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from pyitcast.transformer_utils import SimpleLossCompute
from pyitcast.transformer_utils import run_epoch
from pyitcast.transformer_utils import greedy_decode


def data_generator(V, batch, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        
        data[:, 0] = 1
        
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        
        yield Batch(source, target)


V = 11
batch = 20
num_batch = 30


# if __name__ == '__main__':
#     res = data_generator(V, batch, num_batch)
#     print(res)
        
model = make_model(V, V, N=2)

model_optimizer = get_std_opt(model)

criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0)

loss = SimpleLossCompute(model.generator, criterion, model_optimizer)


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()
        
        run_epoch(data_generator(V, 8, 20), model, loss)
        
        model.eval()
        
        run_epoch(data_generator(V, 8, 5), model, loss)
    
    model.eval()

    source = Variable(torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]]))
    
    source_mask = Variable(torch.ones(1, 1, 10))
    
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == '__main__':
    run(model, loss)





        
        
        
        
        
        





