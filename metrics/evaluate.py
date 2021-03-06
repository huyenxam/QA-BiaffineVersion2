import json
from metrics.f1_score import f1_score
from metrics.exact_match_score import exact_match_score
from dataloader import *

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

        label_prediction = ""
        score_max = 0
        # Lặp qua từng câu trong context
        for ctx in context:
            # mỗi câu bị cắt tương ứng sẽ có 1 điểm số dự đoán của model
            # context[i] <-> outputs[i]
            sentence = ['cls'] + question + ['sep'] +  ctx
            start_pre = outputs[i][1]
            end_pre = outputs[i][2]
            # if start_pre != 0 and end_pre != 0:     # Nếu câu trả lời không phải vị trí của câu negative (0, 0)
            if score_max < outputs[i][3]:       # Nếu điểm số của output cao hơn điểm số max hiện tại thì cập nhật lại điểm số max và vị trí max mới
                    score_max = outputs[i][3]
                    label_prediction = " ".join(sentence[start_pre:end_pre+1])
            i += 1          # Sau mỗi lần lặp của 1 câu thì i tăng thêm 1 đơn vị
        # Lấy câu trả lời trong từng sample
        labels = sample['label']
        f1_idx = [0]
        extract_match_idx = [0]
        for lb in labels:
            ground_truth = lb[3]
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