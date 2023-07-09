import torch
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from Triplet.Model.BERTMRC import BertForQuestionAnswering

#调用模型
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertConfig
from transformers import AdamW
class BertMRCModel(nn.Module):
    def __init__(self, num_labels):
        super(BertMRCModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits



sentence = "I charge it at night and skip taking the cord with me because of the good battery life ."

# 将句子转换为对应的token序列
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(sentence)
input_ids = tokenizer.convert_tokens_to_ids(tokens)


# 设置最大输入长度
max_length = 64

# 进行填充
input_ids = input_ids[:max_length]  # 截断输入，保持最大长度
attention_mask = [1] * len(input_ids)  # 创建attention mask，所有位置设为1
token_type_ids = [0] * len(input_ids)  # 创建token_type_ids，设定为0

padding_length = max_length - len(input_ids)  # 计算填充长度
# print(padding_length)
input_ids += [tokenizer.pad_token_id] * padding_length  # 进行填充
attention_mask += [0] * padding_length  # 更新attention mask
token_type_ids += [0] * padding_length  # 更新token_type_ids


# 将输入转换为PyTorch张量
input_ids = torch.tensor(input_ids).unsqueeze(0)  # 添加batch维度，假设只有一个样本
attention_mask = torch.tensor(attention_mask).unsqueeze(0)
token_type_ids = torch.tensor(token_type_ids).unsqueeze(0)

# 创建模型实例
num_labels = 3  # 情感极性分类的类别数
model = BertMRCModel(num_labels)

#加载训练好的模型参数
model.load_state_dict(torch.load('/Triplet/SaveModel/ASCmodel.pth', map_location='cpu'))
model.eval()  # 设置模型为评估模式

# 使用模型进行预测
with torch.no_grad():  # 关闭梯度计算
    logits = model(input_ids, attention_mask, token_type_ids)

    predicted_indices = torch.argmax(logits, dim=1)  # 获取最大概率对应的索引值

    # 将索引值映射回情感标签
    label_mapping = {
        0: 'POS',
        1: 'NEG',
        2: 'NEU',
    }
    sentiment = label_mapping[predicted_indices.item()]

    print("Sentiment Polarity:", sentiment)
