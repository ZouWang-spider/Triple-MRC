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
