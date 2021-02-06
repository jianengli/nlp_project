# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-chinses')
# model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-chinses')

def get_bert_encode(text1, text2, mark=102, max_len=10):
    indexed_tokens = tokenizer.encode(text1, text2)
    
    sep = indexed_tokens.index(mark)
    
    if len(indexed_tokens[:sep]) >= max_len:
        token1 = indexed_tokens[:max_len]
    else:
        token1 = indexed_tokens[:sep] + (max_len - len(indexed_tokens[:sep])) * [0]
    
    if len(indexed_tokens[sep:]) >= max_len:
        token2 = indexed_tokens[sep:sep + max_len]
    else:
        token2 = indexed_tokens[sep:] + (max_len - len(indexed_tokens[sep:])) * [0]
    
    combine_token = token1 + token2
    
    segment_ids = [0] * max_len + [1] * max_len
    segment_tensor = torch.tensor([segment_ids])
    tokens_tensor = torch.tensor([combine_token])
    
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, token_type_ids=segment_tensor)
        
    return encoded_layers

text_1 = "人生该如何起头"
text_2 = "改变要如何起手"

# encoded_layers = get_bert_encode(text_1, text_2)
# print(encoded_layers)
# print(encoded_layers.shape)


class Net(nn.Module):
    def __init__(self, char_size, embedding_size, dropout=0.2):
        










