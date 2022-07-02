import json
from metrics.f1_score import f1_score
from metrics.exact_match_score import exact_match_score
from dataloader import *
import numpy as np

def evaluate(outputs, path):


    f1 = exact_match = 0 
    list_sample = []        # Lưu trữ danh sách các mẫu
    with open(path, 'r', encoding='utf8') as f:         # Đọc file dataset
        list_sample = json.load(f)

    i = 0               # Lưu trữ chỉ số của từng câu
    # Lặp qua từng mẫu
    for sample in list_sample:
        # Lấy context của từng sample
        context = sample['context']
        question = sample['question'].split(" ")
        text_context = ""
        for item in context:
            text_context += " ".join(item) + " "
        text_context = text_context[:-1].split(' ')

        label_prediction = ""
        score_max = 0
        # Lặp qua từng câu trong context
        for ctx in context:
            # mỗi câu bị cắt tương ứng sẽ có 1 điểm số dự đoán của model
            # context[i] <-> outputs[i]
            sentence = ['cls'] + question + ['sep'] +  ctx
            if score_max < outputs[i][3]:       # Nếu điểm số của output cao hơn điểm số max hiện tại thì cập nhật lại điểm số max và vị trí max mới
                score_max = outputs[i][3]
                start_pre = outputs[i][1]
                end_pre = outputs[i][2]
                label_prediction = " ".join(sentence[start_pre:end_pre+1])
            i += 1          # Sau mỗi lần lặp của 1 câu thì i tăng thêm 1 đơn vị
        # Lấy câu trả lời trong từng sample
        labels = sample['label']
        f1_idx = [0]
        extract_match_idx = [0]
        for lb in labels:
            start = int(lb[1])
            end = int(lb[2])
            ground_truth = " ".join(text_context[start:end+1])
            f1_idx.append(f1_score(label_prediction, ground_truth))
            extract_match_idx.append(exact_match_score(label_prediction, ground_truth))
            # print(ground_truth)
            # print(label_prediction)

        f1 += max(f1_idx)
        exact_match += max(extract_match_idx)    

    total = len(list_sample)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    
    return exact_match, f1