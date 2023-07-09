import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertConfig


#BERT-MRC模型的起始位置和结束位置的输入输出形式
#例如语句：我去吃饭啦！，需要抽取"吃饭"这个词
#输入标签：起始位置是[0, 0, 1, 0, 0, 0]，结束位置[0, 0, 0, 1, 0, 0]   计算loss需要将维度其转为（batch_size,len,2),float类型
#输出标签：起始位置的概率分布为 [0.1, 0.2, 0.4, 0.1, 0.1, 0.1]，对结束位置的概率分布为 [0.1, 0.1, 0.1, 0.5, 0.1, 0.1]
#选取最大概率分布值的位置即能抽取出方面词：吃饭


#对Opinion terms实现序列标注数据处理
def prepare_data(data):
    prepared_data = []
    for item in data:
        text, annotation1,annotation2 = item.split('####')
        text = text.strip()
        # print(annotation2)
        annotation_list = annotation2.strip().split(' ')
        opinions = []
        for annotation in annotation_list:
            if '=' in annotation:
                opinion = annotation.split('=')[1]
                opinions.append(opinion)
        # print(opinions)


        # 寻找答案的起始和结束位置
        start_pos = []
        end_pos = []
        # 捕获 Opinion 标注中的 'S' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('S') and not start_found:
                start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'S' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            end_pos.append(len(opinions) - 1)

        # 捕获 Opinion 标注中的 'SS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SS') and not start_found:
                start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'SS' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
                end_pos.append(len(opinions) - 1)


        # 捕获 Opinion 标注中的 'SSS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SSS') and not start_found:
                start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'SSS' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            end_pos.append(len(opinions) - 1)

        # 捕获 Opinion 标注中的 'SSSS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SSSS') and not start_found:
                start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'SSSS' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            end_pos.append(len(opinions) - 1)


        # 捕获 Opinion 标注中的 'SSSSS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SSSSS') and not start_found:
                start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'SSSSS' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            end_pos.append(len(opinions) - 1)


        # 捕获 Opinion 标注中的 'SSSSSS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SSSSSS') and not start_found:
                start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'TTTTTT-' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            end_pos.append(len(opinions) - 1)


        #根据抽取的aspect terms处理question
        from Triplet.DataProcess.AspectProcess import prepare_aspect
        aspect_start_pos,aspect_end_pos=prepare_aspect(data)

        aspect_terms = extract_aspect_terms(text, aspect_start_pos,aspect_end_pos)     #start_pos, end_pos需要用Aspect Extraction的
        # print(aspect_terms)

        # 构建问题  question中的aspect terms是AE模型输出的结果
        # question = "Find the opinion terms for aspect terms in the text."
        for i, aspect_term in enumerate(aspect_terms):
            question = "Find the opinion terms for {aspect term} in the text."
            question = question.replace("{aspect term}", aspect_term)
            start=start_pos[i]
            # print(start)
            end=end_pos[i]

            prepared_data.append({
                'context': text,
                'question': question,
                'answer_start': start,
                'answer_end': end
            })


    return prepared_data


#根据句子，开始位置，结束位置抽取aspect terms
def extract_aspect_terms(sentence, start_positions, end_positions):
    aspect_terms = []
    words = sentence.split()
    for start, end in zip(start_positions, end_positions):
        term = " ".join(words[start:end + 1])
        aspect_terms.append(term)

    return aspect_terms

#数据集加载处理
def load_data(data_file):
    prepared_data = []
    with open(data_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
          data_list = line.strip()
          prepared = prepare_data([data_list]) # 将数据列表传递给prepare_data函数
          prepared_data.extend(prepared)  # 将处理后的结果添加到prepared_data中
    return prepared_data

# 数据集文件路径
data_file = 'E:/Pythonproject/Triplet/Data/16res/16rest_train.txt'
# 加载数据
data = load_data(data_file)
# print(data)


#对数据进行编码，适合BERT-MRC模型的输入格式。
def MRCDataset(data):
    encoded_data = []
    for index in range(len(data)):
        # 获取样本
        sample = data[index]

        # 获取文本、问题和答案
        context = sample['context']
        question = sample['question']
        start_pos = sample['answer_start']
        end_pos = sample['answer_end']

        # 对文本和问题进行编码
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')

        # 获取特征表示和目标标签
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs['token_type_ids'].squeeze()
        start_positions = torch.tensor(start_pos, dtype=torch.long)
        end_positions = torch.tensor(end_pos, dtype=torch.long)

        encoded_data.append( {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'start_positions': start_positions,
            'end_positions': end_positions
        })
    return encoded_data


# 加载预训练的BERT模型和分词器
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#序列填充处理
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    start_positions = [item['start_positions'] for item in batch]
    end_positions = [item['end_positions'] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True)
    padded_token_type_ids = pad_sequence(token_type_ids, batch_first=True)

    # 计算最大长度
    max_length = padded_input_ids.size(1)
    # 填充start_positions,针对多个aspect terms对应多个opinion terms的起始位置和结束位置
    # padded_start_positions = torch.zeros((len(start_positions),max_length), dtype=torch.long)
    # for i, pos_list in enumerate(start_positions):
    #     for pos in pos_list:
    #         if pos != 0:
    #             padded_start_positions[i, pos] = 1
    # # 填充start_positions
    # padded_end_positions = torch.zeros((len(end_positions),max_length ), dtype=torch.long)
    # for i, pos_list in enumerate(end_positions):
    #     for pos in pos_list:
    #         if pos != 0:
    #             padded_end_positions[i, pos] = 1

    # 填充start_positions
    padded_start_positions = torch.zeros((len(start_positions), max_length), dtype=torch.long)
    for i, pos in enumerate(start_positions):
        if pos != 0:
            padded_start_positions[i, pos] = 1

    # 填充end_positions
    padded_end_positions = torch.zeros((len(end_positions), max_length), dtype=torch.long)
    for i, pos in enumerate(end_positions):
        if pos != 0:
            padded_end_positions[i, pos] = 1

    return padded_input_ids, padded_attention_mask,padded_token_type_ids, padded_start_positions, padded_end_positions


# 加载并准备训练数据
train_dataset = MRCDataset(data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn)


#m构建BERT-MRC模型
class BertMRCModel(nn.Module):
    def __init__(self, config):
        super(BertMRCModel, self).__init__()
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
model = BertMRCModel(config)  # 创建模型实例
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # 定义优化器
reduction = 'none'
Epoch=50
loss = nn.CrossEntropyLoss(reduction=reduction)
# criterion2 = LabelSmoothingCrossEntropy(reduction=reduction)
# criterion3 = FocalLoss(reduction=reduction)
loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
SavePath = "E:/Pythonproject/Triple/SaveModel/OEmodel.pth"

#计算Span损失值
def spanloss(start_logits,end_logits,start_positions,end_positions):

    # 得到start_probs和end_probs
    start_probs = F.softmax(start_logits, dim=1)
    end_probs = F.softmax(end_logits, dim=1)

    # 计算标签的分布得到target_probs
    target_probs = torch.zeros_like(start_probs)
    start_positions = start_positions.unsqueeze(-1)  # Add a new dimension at the end
    target_probs.scatter_(1, start_positions, 1)
    end_positions = end_positions.unsqueeze(-1)  # Add a new dimension at the end
    target_probs.scatter_(1, end_positions, 1)

    # 损失函数计算start_probs和end_probs与target_probs之间的差值
    loss_fn = nn.CrossEntropyLoss()
    start_loss = loss_fn(start_probs, target_probs)
    end_loss = loss_fn(end_probs, target_probs)
    sopan_loss = start_loss + end_loss
    return sopan_loss


# 训练BERT-MRC模型
for epoch in range(Epoch):
    print("训练迭代的次数",epoch)
    model.train()  # 进入训练模式

    # 遍历数据批次
    for batch in train_loader:
        input_ids, attention_mask, token_type_ids, start_positions, end_positions = batch
        # print('标签的真实值',start_positions.size())   #32,len

        optimizer.zero_grad()  # 清零梯度
        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)  # 前向传播
        # print('预测的概率分布',start_logits.size())    #32,len，2

        span_loss = spanloss(start_logits, end_logits, start_positions, end_positions)

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
        total_loss = start_loss + end_loss #+ span_loss
        total_loss = torch.mean(total_loss)

        total_loss.backward()
        optimizer.step()
        # 打印批次的损失
        print("Batch Loss", total_loss.item())


    # 在所有的epoch训练完成后
    if epoch == Epoch-1:
        # 保存模型的权重
        torch.save(model.state_dict(), SavePath)
        print("模型权重已保存")





