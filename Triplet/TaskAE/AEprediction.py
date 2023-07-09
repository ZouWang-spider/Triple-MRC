import torch
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence



sentence = "I charge it at night and skip taking the cord with me because of the good battery life ."

# 将句子转换为对应的token序列
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(sentence)
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# 设置最大输入长度
max_length = 128

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



#调用模型加载参数进行预测
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertConfig

class BertForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = BertModel(config)
        self.mid_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.start_fc = nn.Linear(config.hidden_size, 2)
        self.end_fc = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        seq_out = bert_outputs[0]
        seq_out = self.mid_linear(seq_out)
        start_logits = self.start_fc(seq_out)
        end_logits = self.end_fc(seq_out)
        return start_logits, end_logits


config = BertConfig()  # 使用默认的BERT配置
model = BertForQuestionAnswering(config)  # 创建模型实例
#加载训练好的模型参数
model.load_state_dict(torch.load('/Triplet/SaveModel/AEmodel2.pth', map_location='cpu'))
model.eval()  # 设置模型为评估模式


# 使用模型进行预测
with torch.no_grad():  # 关闭梯度计算
    start_logits,end_logits = model(input_ids, attention_mask, token_type_ids)
    # print('预测开始位置的概率分布：',start_logits)
    # print('预测结束位置的概率分布：', start_logits)

    # 截取与句子相关的概率分布
    start_logits = start_logits[0, :len(tokens)]
    # print(start_logits)
    end_logits = end_logits[0, :len(tokens)]


    start_index = torch.argmax(start_logits)
    # print(start_index)
    end_index = torch.argmax(end_logits)
    # print(end_index)

    # 确保开始位置和结束位置的合理性
    if start_index > end_index:
        aspect_term = None
        print('预测结果不合理')
    else:
        # 提取 aspect term
        aspect_term = " ".join(tokens[start_index:end_index + 1])

    print("Aspect Terms:", aspect_term)