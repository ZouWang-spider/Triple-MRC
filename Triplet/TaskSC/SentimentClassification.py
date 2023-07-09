import torch
import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


#数据集预处理
#获取序列标注中每个方面词的情感极性
def prepare_data(data):
    prepared_data = []
    for item in data:
        text, annotation1,annotation2 = item.split('####')
        text = text.strip()
        annotation_list = annotation1.strip().split(' ')

        # 确定句子中aspect terms的情感极性
        sentiments = []
        for i, annotation in enumerate(annotation_list):

            # 获取T-开头情感极性
            if '=T-' in annotation:
                sentiment = annotation.split('=T-')[1]
                #判断是否为多词方面词
                if i > 0 and (annotation_list[i-1].endswith('=O') or i == 0):
                   sentiments.append(sentiment)


            # 获取TT-开头情感极性
            if '=TT-' in annotation:
                sentiment = annotation.split('=TT-')[1]
                # 判断是否为多词方面词
                if i > 0 and (annotation_list[i - 1].endswith('=O') or i == 0):
                  sentiments.append(sentiment)


            # 获取TTT-开头情感极性
            if '=TTT-' in annotation:
                sentiment = annotation.split('=TTT-')[1]
                # 判断是否为多词方面词
                if i > 0 and (annotation_list[i - 1].endswith('=O') or i == 0):
                    sentiments.append(sentiment)


            # 获取TTTT-开头情感极性
            if '=TTTT-' in annotation:
                sentiment = annotation.split('=TTTT-')[1]
                # 判断是否为多词方面词
                if i > 0 and (annotation_list[i - 1].endswith('=O') or i == 0):
                    sentiments.append(sentiment)


            # 获取TTTTT-开头情感极性
            if '=TTTTT-' in annotation:
                sentiment = annotation.split('=TTTTT-')[1]
                # 判断是否为多词方面词
                if i > 0 and (annotation_list[i - 1].endswith('=O') or i == 0):
                    sentiments.append(sentiment)


            # 获取TTTTTT-开头情感极性
            if '=TTTTTT-' in annotation:
                sentiment = annotation.split('=TTTTTT-')[1]
                # 判断是否为多词方面词
                if i > 0 and (annotation_list[i - 1].endswith('=O') or i == 0):
                    sentiments.append(sentiment)


        #根据抽取的aspect terms处理question
        from Triplet.DataProcess.AspectProcess import prepare_aspect
        aspect_start_pos,aspect_end_pos=prepare_aspect(data)
        aspect_terms = extract_terms(text, aspect_start_pos,aspect_end_pos)
        # print(aspect_terms)  #['battery life']

        # 根据抽取的opinion terms处理question
        from Triplet.DataProcess.OpinionProcess import prepare_opinion
        aspect_start_pos, aspect_end_pos = prepare_opinion(data)
        opinion_terms = extract_terms(text, aspect_start_pos, aspect_end_pos)
        # print(opinion_terms)   #['good']


        # 构建问题  question中的aspect terms是AE模型输出的结果
        # question = "Find the sentiment polarity of the {opinion terms} related to the {aspect terms} in the text."
        for opinion_term, aspect_term, sentiment in zip(opinion_terms, aspect_terms,sentiments):
            question = "Find the sentiment polarity of the {opinion terms} related to the {aspect terms} in the text."
            question = question.replace("{opinion terms}", opinion_term).replace("{aspect terms}", aspect_term)


            prepared_data.append({
                'context': text,
                'question': question,
                'sentiment':sentiment
            })


    return prepared_data


#根据句子，开始位置，结束位置抽取aspect terms
def extract_terms(sentence, start_positions, end_positions):
    terms = []
    words = sentence.split()
    for start, end in zip(start_positions, end_positions):
        term = " ".join(words[start:end + 1])
        terms.append(term)

    return terms

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

# 定义情感标签映射
label_mapping = {
    'POS': 0,
    'NEG': 1,
    'NEU': 2,
}

#对数据进行编码，适合BERT-MRC（ASC)模型的输入格式。
def MRCDataset(data):
    encoded_data = []
    for index in range(len(data)):
        # 获取样本
        sample = data[index]

        # 获取文本、问题和答案
        context = sample['context']
        question = sample['question']
        sentiment = sample['sentiment']

        # 对文本和问题进行编码
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors='pt')

        # 对情感标签进行编码
        sentiment_label = label_mapping[sentiment]

        # 获取特征表示和目标标签
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs['token_type_ids'].squeeze()

        encoded_data.append( {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'sentiment_label': sentiment_label
        })
    return encoded_data

# 加载预训练的BERT模型和分词器
# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#对数据进行填充
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    sentiment_labels = [item['sentiment_label'] for item in batch]

    padded_input_ids = pad_sequence(input_ids, batch_first=True)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True)
    padded_token_type_ids = pad_sequence(token_type_ids, batch_first=True)


    return padded_input_ids, padded_attention_mask,padded_token_type_ids,sentiment_labels

# 加载并准备训练数据
train_dataset = MRCDataset(data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn)


#构建模型
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


# model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

num_labels = 3  # 情感极性分类的类别数
model = BertMRCModel(num_labels)
#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs=50
SavePath="E:/Pythonproject/Triple/SaveModel/ASCmodel.pth"

#模型训练
for epoch in range(num_epochs):
    print('训练的次数：',epoch)
    for batch in train_loader:
        input_ids, attention_mask, token_type_ids, sentiment_labels = batch
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        sentiment_labels_tensor = torch.tensor(sentiment_labels)   #将list数据转化为Tensor对象
        loss = criterion(logits, sentiment_labels_tensor)
        loss.backward()
        optimizer.step()
        # 打印批次的损失
        print("Batch Loss", loss.item())

    # 在所有的epoch训练完成后
    if epoch == num_epochs - 1:
        # 保存模型的权重
        torch.save(model.state_dict(), SavePath)
        print("模型权重已保存")


# # 评估循环
# model.eval()
# with torch.no_grad():
#     for input_ids, attention_mask, token_type_ids, sentiment_labels in test_loader:
#         logits = model(input_ids, attention_mask, token_type_ids)
#         predicted_labels = torch.argmax(logits, dim=1)
#         # 计算准确率、精确率、召回率等评估指标