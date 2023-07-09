#对Opinion terms实现序列标注数据处理

def prepare_opinion(data):
    for item in data:
        text, annotation1,annotation2 = item.split('####')
        annotation_list = annotation2.strip().split(' ')
        opinions = []
        for annotation in annotation_list:
            if '=' in annotation:
                opinion = annotation.split('=')[1]
                opinions.append(opinion)


        # 寻找答案的起始和结束位置
        opinion_start_pos = []
        opinion_end_pos = []
        # 捕获 Opinion 标注中的 'S' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('S') and not start_found:
                opinion_start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                opinion_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'S' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            opinion_end_pos.append(len(opinions) - 1)

        # 捕获 Opinion 标注中的 'SS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SS') and not start_found:
                opinion_start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                opinion_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'SS' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
                opinion_end_pos.append(len(opinions) - 1)


        # 捕获 Opinion 标注中的 'SSS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SSS') and not start_found:
                opinion_start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                opinion_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'SSS' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            opinion_end_pos.append(len(opinions) - 1)

        # 捕获 Opinion 标注中的 'SSSS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SSSS') and not start_found:
                opinion_start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                opinion_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'SSSS' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            opinion_end_pos.append(len(opinions) - 1)


        # 捕获 Opinion 标注中的 'SSSSS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SSSSS') and not start_found:
                opinion_start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                opinion_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'SSSSS' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            opinion_end_pos.append(len(opinions) - 1)


        # 捕获 Opinion 标注中的 'SSSSSS' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(opinions)):
            if opinions[i].startswith('SSSSSS') and not start_found:
                opinion_start_pos.append(i)
                start_found = True
            if opinions[i] == 'O' and start_found:
                opinion_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'TTTTTT-' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            opinion_end_pos.append(len(opinions) - 1)

        return opinion_start_pos,opinion_end_pos