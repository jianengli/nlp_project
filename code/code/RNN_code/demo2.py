import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, embedding_size, num_layers, batch_size, vocab_size,
                 tag_to_id, hidden_size, sentence_length, batch_first):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.tag_to_id = tag_to_id
        self.hidden_size = hidden_size // 2
        self.output_size = len(tag_to_id)
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.sentence_length = sentence_length
        self.batch_first = batch_first
    
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        
        self.bilstm = nn.LSTM(embedding_size, self.hidden_size, num_layers, 
                              bidirectional=True, batch_first=batch_first)
        
        self.linear = nn.Linear(hidden_size, self.output_size)
        
        
    def forward(self, sentence_sequences):
        h_0 = torch.randn(self.num_layers * 2, self.batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_layers * 2, self.batch_size, self.hidden_size)
        # print(h_0.shape)  [2, 8, 64]
        
        input_features = self.embedding(sentence_sequences)
        
        # print(sentence_sequences.shape)  [8, 20]
        # print(input_features.shape)      [8, 20, 200]
        
        # LSTM的输入要求是[batch_size, seq_length, hidden_size], 前提是batch_first=True
        output, (h_n, c_n) = self.bilstm(input_features, (h_0, c_0))
        
        # print(output.shape)  [8, 20, 128]
        # print(h_n.shape)     [2, 8, 64]
        
        output_features = self.linear(output)
        # print(output_features.shape)  [8, 20, 5]
        
        return output_features
    

def sentence_map(sentence_list, char_to_id, max_length):
    sentence_list.sort(key=lambda x: len(x), reverse=True)
    
    sentence_map = []
    
    for sen in sentence_list:
        sentence_id_list = [char_to_id[c] for c in sen]
        
        padding = [0] * (max_length - len(sentence_id_list))
        
        sentence_id_list.extend(padding)
        
        sentence_map.append(sentence_id_list)
        
    return torch.tensor(sentence_map, dtype=torch.long)


if __name__ == '__main__':
    sentence_list = ["确诊弥漫大b细胞淋巴瘤1年", "反复咳嗽、咳痰40年,再发伴气促5天。",
        "生长发育迟缓9年。", "右侧小细胞肺癌第三次化疗入院",
        "反复气促、心悸10年,加重伴胸痛3天。", "反复胸闷、心悸、气促2多月,加重3天",
        "咳嗽、胸闷1月余, 加重1周", "右上肢无力3年, 加重伴肌肉萎缩半年"]
    
    char_to_id = {'<PAD>':0}
    for sen in sentence_list:
        for c in sen:
            if c not in char_to_id:
                char_to_id[c] = len(char_to_id)
    
    tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}
    
    sentence_map_list = sentence_map(sentence_list, char_to_id, 20)
    
    # print(sentence_map_list.shape)  [8, 20]
    
    model = BiLSTM(200, 1, 8, len(char_to_id), tag_to_id, 128, 100, True)
    
    # print(sentence_map_list)
    
    result = model(sentence_map_list)
    # print(result)
    # print(result.shape)  [8, 20, 5]
    # [8, 20, 5] - [batch_size, max_length, output_size]
    # 其中最后一维output_size, 是用len(tag_to_id)求出来的



'''
if __name__ == '__main__':
    char_to_id = {"双": 0, "肺": 1, "见": 2, "多": 3, "发": 4, "斑": 5, "片": 6,
                  "状": 7, "稍": 8, "高": 9, "密": 10, "度": 11, "影": 12, "。": 13}

    tag_to_id = {"O": 0, "B-dis": 1, "I-dis": 2, "B-sym": 3, "I-sym": 4}

    EMBEDDING_DIM = 200

    HIDDEN_DIM = 128

    BATCH_SIZE = 8

    SENTENCE_LENGTH = 100

    NUM_LAYERS = 1

    model = BiLSTM(200, 1, 8, len(char_to_id), tag_to_id, 128, 100, True)
    print(model)
'''

