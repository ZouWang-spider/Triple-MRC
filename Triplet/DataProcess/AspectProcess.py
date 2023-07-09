#对Aspect terms实现序列标注数据处理
def prepare_aspect(data):
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

        # 寻找答案的起始和结束位置
        aspect_start_pos = []
        aspect_end_pos = []

        # 捕获 Aspect 标注中的 'T-' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('T-') and not start_found:
                aspect_start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                aspect_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'T-POS' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            aspect_end_pos.append(len(aspects) - 1)

        # 捕获 Aspect 标注中的 'TT-' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TT-') and not start_found:
                aspect_start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                aspect_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'TT-' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
                aspect_end_pos.append(len(aspects) - 1)


        # 捕获 Aspect 标注中的 'TTT-' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TTT-') and not start_found:
                aspect_start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                aspect_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'TTT-' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            aspect_end_pos.append(len(aspects) - 1)


        # 捕获 Aspect 标注中的 'TTTT-' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TTTT-') and not start_found:
                aspect_start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                aspect_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'TTTT-' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            aspect_end_pos.append(len(aspects) - 1)


        # 捕获 Aspect 标注中的 'TTTTT-' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TTTTT-') and not start_found:
                aspect_start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                aspect_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'TTTTT-' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            aspect_end_pos.append(len(aspects) - 1)


        # 捕获 Aspect 标注中的 'TTTTTT-' 的开始位置和结束位置
        start_found = False
        end_found = False
        for i in range(len(aspects)):
            if aspects[i].startswith('TTTTTT-') and not start_found:
                aspect_start_pos.append(i)
                start_found = True
            if aspects[i] == 'O' and start_found:
                aspect_end_pos.append(i - 1)
                end_found = True
                break
        # 如果 'TTTTTT-' 在最后一个位置上，则将其添加为结束位置
        if not end_found and start_found:
            aspect_end_pos.append(len(aspects) - 1)


        return aspect_start_pos,aspect_end_pos

