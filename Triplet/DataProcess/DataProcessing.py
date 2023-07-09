#Datapreprocessing processed the datasets provided by Peng et al (2020,AAAI) into the BERT-MRC model input form
#The dataset includes text, sequential annotation of aspect words (joint tagging, T-POS, TT-POS, TTT-POS/NEG), viewpoint word annotation (S,SS,SSS,...)
#The data set is processed context, question, start_position, and end_position
#such as :
# context:Its ease of use and the top service from Apple- be it their phone assistance or bellying up to the genius bar -- can not be beat .
####Its=O ease=O of=O use=T-POS and=O the=O top=O service=TT-POS from=O Apple-=O be=O it=O their=O phone=TTT-POS assistance=TTT-POS or=O bellying=O up=O to=O the=O genius=TTTT-POS bar=TTTT-POS --=O can=O not=O be=O beat=O .=O
####Its=O ease=S of=O use=O and=O the=O top=SS service=O from=O Apple-=O be=O it=O their=O phone=O assistance=O or=O bellying=O up=O to=O the=O genius=O bar=O --=O can=SSS not=SSS be=SSS beat=SSS .=O

import re
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


#Sequence annotation for Aspect terms
def prepare_data(data):
    prepared_data = []
    for item in data:
        text, annotation1,annotation2 = item.split('####')
        text = text.strip()
        annotation_list = annotation1.strip().split(' ')
        aspects = []
        for annotation in annotation_list:
            if '=' in annotation:
                aspect = annotation.split('=')[1]
                aspects.append(aspect)
        # print(aspects)

        # Building the Question
        question = "Find the aspect terms in the text."


        # Find the start_position and end_position of the Answer
        start_pos = []
        end_pos = []

        # Capture the start_position and end_position of the 'T-' in the Aspect Terms
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('T-') and not start_found:
                start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break
        # If 'T-POS' is in the last_position, add it as the end_position
        if not end_found and start_found:
            end_pos.append(len(aspects) - 1)

        # Capture the start_position and end_position of the 'TT-' in the Aspect Terms
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TT-') and not start_found:
                start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break

        if not end_found and start_found:
                end_pos.append(len(aspects) - 1)


        # Capture the start_position and end_position of the 'TTT-' in the Aspect Terms
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TTT-') and not start_found:
                start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break

        if not end_found and start_found:
            end_pos.append(len(aspects) - 1)


        # Capture the start_position and end_position of the 'TTTT-' in the Aspect Terms
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TTTT-') and not start_found:
                start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break

        if not end_found and start_found:
            end_pos.append(len(aspects) - 1)


        # Capture the start_position and end_position of the 'TTTTT-' in the Aspect Terms
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TTTTT-') and not start_found:
                start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break

        if not end_found and start_found:
            end_pos.append(len(aspects) - 1)


        # Capture the start_position and end_position of the 'TTTTTT-' in the Aspect Terms
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TTTTTT-') and not start_found:
                start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                end_pos.append(i - 1)
                end_found = True
                break

        if not end_found and start_found:
            end_pos.append(len(aspects) - 1)

        prepared_data.append({
            'context': text,
            'question': question,
            'answer_start': start_pos,
            'answer_end': end_pos
        })

    return prepared_data

#Datasets loading and processing
def load_data(data_file):
    prepared_data = []
    with open(data_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
          data_list = line.strip()
          prepared = prepare_data([data_list])
          prepared_data.extend(prepared)
    return prepared_data

# Dataset file path
data_file = '/Triplet/Data/14lap/train.txt'
# Loading dataset
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
    # 填充start_positions
    padded_start_positions = torch.zeros((len(start_positions),max_length), dtype=torch.long)
    for i, pos_list in enumerate(start_positions):
        for pos in pos_list:
            if pos != 0:
                padded_start_positions[i, pos] = 1
    # 填充start_positions
    padded_end_positions = torch.zeros((len(end_positions),max_length ), dtype=torch.long)
    for i, pos_list in enumerate(end_positions):
        for pos in pos_list:
            if pos != 0:
                padded_end_positions[i, pos] = 1


    return padded_input_ids, padded_attention_mask,padded_token_type_ids, padded_start_positions, padded_end_positions

# 加载并准备训练数据
train_dataset = MRCDataset(data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,collate_fn=collate_fn)