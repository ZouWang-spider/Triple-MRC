from transformers import BertModel, BertTokenizer
from Triplet.Model.BERTMRC import BertMRCModel
import torch
import torch.nn as nn
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
from transformers import BertConfig
from Triplet.TaskAE.AspectExtraction import MRCDataset as MRCDataset1, load_data, collate_fn as collate_fn1
from Triplet.TaskOE.OpinionExtraction import MRCDataset as MRCDataset2, collate_fn as collate_fn2
from Triplet.TaskSC.SentimentClassification import MRCDataset as MRCDataset3, collate_fn as collate_fn3

# 加载预训练的BERT模型和分词器
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)



# 创建输入文本
text1 = "Aspect Extraction Input"
text2 = "Opinion Extraction Input"
text3 = "Aspect term Sentimrnt Analysis Input"

# 使用分词器对文本进行编码
inputs1 = tokenizer.encode_plus(text1, add_special_tokens=True, return_tensors='pt')
inputs2 = tokenizer.encode_plus(text2, add_special_tokens=True, return_tensors='pt')
inputs3 = tokenizer.encode_plus(text2, add_special_tokens=True, return_tensors='pt')

# 获取输入的张量
def collate_fn1(batch):
   collate_fn1(batch)  #AE

def collate_fn2(batch):
    collate_fn2(batch)  # OE

def collate_fn3(batch):
     collate_fn3(batch)  # AE


data_file = '/Triplet/Data/14lap/train.txt'
data = load_data(data_file)
train_dataset1 = MRCDataset1(data)
train_dataset2 = MRCDataset2(data)
train_dataset3 = MRCDataset3(data)
train_loader1 = DataLoader(train_dataset1, batch_size=32, shuffle=True,collate_fn=collate_fn1)
train_loader2 = DataLoader(train_dataset2, batch_size=32, shuffle=True,collate_fn=collate_fn2)
train_loader3 = DataLoader(train_dataset3, batch_size=32, shuffle=True,collate_fn=collate_fn3)


# 使用BERT模型进行前向传播
config = BertConfig()  # 使用默认的BERT配置
model = BertMRCModel.from_pretrained(config)  # 创建模型实例
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # 定义优化器
reduction = 'none'
Epoch=10
loss = nn.CrossEntropyLoss(reduction=reduction)
# criterion2 = LabelSmoothingCrossEntropy(reduction=reduction)
# criterion3 = FocalLoss(reduction=reduction)
loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
SavePath = "E:/Pythonproject/Triple/SaveModel/Sharedmodel.pth"


# 循环训练
for epoch in range(Epoch):
    print("训练迭代的次数",epoch)
    model.train()  # 进入训练模式

    # 遍历数据批次
    for batch in train_loader1:
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch
        # print('标签的真实值',start_positions.size())   #32,len

        optimizer.zero_grad()  # 清零梯度
        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)  # 前向传播
        # print('预测的概率分布',start_logits.size())    #32,len，2


        start_positions = start_positions.unsqueeze(-1)
        start_positions = start_positions.expand(-1, -1, 2)
        start_positions = start_positions.float()
        # print('标签的真实值', start_positions.size())
        end_positions = end_positions.unsqueeze(-1)
        end_positions = end_positions.expand(-1, -1, 2)
        end_positions = end_positions.float()

        # 计算损失函数
        start_loss = loss(start_logits, start_positions)
        end_loss = loss(end_logits, end_positions)
        total_loss = start_loss + end_loss  #+ span_loss
        total_loss = torch.mean(total_loss)

        total_loss.backward()
        optimizer.step()
        # 打印批次的损失
        print("Batch Loss", total_loss.item())


