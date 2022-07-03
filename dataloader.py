import torch
import json
import numpy as np
from torch.utils.data import Dataset


class InputSample(object):
    def __init__(self, path, max_char_len=None, max_seq_length=None):
        self.max_char_len = max_char_len        # Độ dài tối đa ký tự của một từ
        self.max_seq_length = max_seq_length    # Độ dài tối đa của context
        self.list_sample = []                   # Danh sách các mẫu
        with open(path, 'r', encoding='utf8') as f:     # Đọc file dataset
            self.list_sample = json.load(f)
        # self.list_sample = self.list_sample[:10]
        
    def get_character(self, word, max_char_len):
        word_seq = []
        for j in range(max_char_len):
            try:
                char = word[j]
            except:
                char = 'PAD'
            word_seq.append(char)
        return word_seq

    def get_sample(self):
        l_sample = []
        for i, sample in enumerate(self.list_sample):               # Lặp qua từng sample
            text_question = sample['question'].split(' ')           # Tách question thành từng từ
            
            context = sample['context']                             #
            text_context = ""                                       #
            for item in context:                                    # Tách context thành một list các từ
              text_context += " ".join(item) + " "                  #
            text_context = text_context[:-1].split(' ')             #

            sent = text_question + text_context                             #
            char_seq = []                                                   #
            for word in sent:                                               # char_seq chứa danh sách cac ký tự cho từng từ
                character = self.get_character(word, self.max_char_len)     #
                char_seq.append(character)                                  #

            len_ctx = 0
            for ctx in context:         # Lặp qua từng câu trong context
                qa_dict = {}            # Khởi tạo từ điển các mẫu

                length_ctx = self.max_seq_length - len(text_question) - 2   # Độ dài tối đa của một câu
                if len(ctx) > length_ctx:               # Nếu độ dài của một câu vượt quá độ dài tối đa thì cắt bớt cho = với độ đài cho phép
                  ctx = ctx[:length_ctx]
                
                labels = sample['label']
                label_list = []         # Khởi tạo danh sách các nhãn
                for lb in labels:       # Lặp qua từng nhãn
                    entity = lb[0]
                    start = int(lb[1])
                    end = int(lb[2])

                    start_ctx = 0
                    end_ctx = 0
                    if start >= len_ctx and end <= (len_ctx + len(ctx) - 1):        # Nếu nhãn thuộc trong câu được xét thì lưu lại trong label_list
                        start_ctx = start - len_ctx + len(text_question) + 2
                        end_ctx = end - len_ctx + len(text_question) + 2
                        if end_ctx >= self.max_seq_length:      # Nếu câu trả lời vượt quá độ dài câu thì cắt bỏ
                            end_ctx = self.max_seq_length - 1

                    label_list.append([entity, start_ctx, end_ctx])
                    # Nếu không có câu trả lời nào thuộc câu thì label_list = ["ANSWER", 0 , 0]
                qa_dict['question'] = text_question     # Lưu trữ lại question 
                qa_dict['context'] = ctx                # Lưu trữ lại câu
                qa_dict['char_sequence'] = char_seq     # Lưu trữ lại ký tự
                qa_dict['label_idx'] = label_list       # Lưu trữ lại label
                len_ctx = len_ctx + len(ctx)

                l_sample.append(qa_dict)

        return l_sample


class MyDataSet(Dataset):

    def __init__(self, path, char_vocab_path, label_set_path,
                 max_char_len, tokenizer, max_seq_length):

        self.samples = InputSample(path=path, max_char_len=max_char_len, max_seq_length=max_seq_length).get_sample()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_char_len = max_char_len
        with open(label_set_path, 'r', encoding='utf8') as f:
            self.label_set = f.read().splitlines()

        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)
        self.label_2int = {w: i for i, w in enumerate(self.label_set)}

    def preprocess(self, tokenizer, context, question, max_seq_length, mask_padding_with_zero=True):
        input_ids = [tokenizer.cls_token_id]            # Thêm [CLS] vào đầu câu
        firstSWindices= [len(input_ids)]

        for w in question:
            word_token = tokenizer.encode(w)                    # Chuyển các token thành số
            input_ids += word_token[1: (len(word_token) - 1)]   # Chỉ lấy token đầu tiên
                                                                # Example: seq = "Chúng tôi"
                                                                # tokenizer.encode("Chúng tôi") -> [0, 746, 2]
                                                                # Lấy token đầu tiên tại vị trí [1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))               # lưu lại vị trí token đã lấy 
        
        input_ids.append(tokenizer.sep_token_id)                # Thêm [SEP] và giữa question và context
        firstSWindices.append(len(input_ids))

        for w in context:
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            if len(input_ids) >= max_seq_length:
              firstSWindices.append(0)
            else:
              firstSWindices.append(len(input_ids))

        input_ids.append(tokenizer.sep_token_id)
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        if len(firstSWindices) > max_seq_length:
            firstSWindices = firstSWindices[:max_seq_length]
        else:
            firstSWindices = firstSWindices + [0] * (max_seq_length - len(firstSWindices))

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]
        else:
            attention_mask = attention_mask + [0 if mask_padding_with_zero else 1] * (max_seq_length - len(input_ids))
            input_ids = (
                    input_ids
                    + [
                        tokenizer.pad_token_id,
                    ]
                    * (max_seq_length - len(input_ids))
            )

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(firstSWindices)

    def character2id(self, character_sentence, max_seq_length):
        char_ids = []
        for word in character_sentence:
            word_char_ids = []
            for char in word:
                if char not in self.char_vocab:
                    word_char_ids.append(self.char_vocab['UNK'])
                else:
                    word_char_ids.append(self.char_vocab[char])
            char_ids.append(word_char_ids)
        if len(char_ids) < max_seq_length:
            char_ids += [[self.char_vocab['PAD']] * self.max_char_len] * (max_seq_length - len(char_ids))
        else:
            char_ids = char_ids[:max_seq_length]
        return torch.tensor(char_ids)

    def span_maxtrix_label(self, label):
        start, end, entity = [], [], []
        label = np.unique(label, axis=0).tolist()       # Loại bỏ những label trùng nhau
        for lb in label:                                # lặp qua từng label
            if int(lb[1]) > self.max_seq_length or int(lb[2]) > self.max_seq_length:        # Nếu vị trí bd hoặc kết thúc lớn hơn max_seq_length thì chuyển thành vị trí (0, 0)
                start.append(0)
                end.append(0)
            else:
                start.append(int(lb[1]))
                end.append(int(lb[2]))
            try:
                entity.append(self.label_2int[lb[0]])
            except:
                print(lb)
        
        label = torch.sparse.FloatTensor(torch.tensor([start, end], dtype=torch.int64), torch.tensor(entity),
                                         torch.Size([self.max_seq_length, self.max_seq_length])).to_dense()
        
        # Example: start = 2, end = 3, entity = 1, max_seq_length = 5
        """ label = tensor([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
            label.shape = [max_seq_length, max_seq_length]"""
        # Example: start = [], end = [], entity = [], max_seq_length = 5
        """ label = tensor([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])
            label.shape = [max_seq_length, max_seq_length]"""
        return label

    def __getitem__(self, index):

        sample = self.samples[index]
        context = sample['context']
        question = sample['question']
        char_seq = sample['char_sequence']
        seq_length = len(question) + len(context) + 2        
        label = sample['label_idx']
        input_ids, attention_mask, firstSWindices = self.preprocess(self.tokenizer, context, question, self.max_seq_length)

        char_ids = self.character2id(char_seq, max_seq_length=self.max_seq_length)
        if seq_length > self.max_seq_length:
          seq_length = self.max_seq_length
        label = self.span_maxtrix_label(label)

        return input_ids, attention_mask, firstSWindices, torch.tensor([seq_length]), char_ids, label.long()

    def __len__(self):
        return len(self.samples)


def get_mask(max_length, seq_length):
    mask = [[1] * seq_length[i] + [0] * (max_length - seq_length[i]) for i in range(len(seq_length))]
    mask = torch.tensor(mask)
    mask = mask.unsqueeze(1).expand(-1, mask.shape[-1], -1)
    mask = torch.triu(mask)

    # Example: seq_length = [2], max_length = 5
    """ mask = tensor([[[1, 1, 1, 1, 0],        # start_seq = 0 , end_seq = seq_length
                        [0, 1, 1, 1, 0],        # start_seq = 1 , end_seq = seq_length
                        [0, 0, 1, 1, 0],        # start_seq = 2 , end_seq = seq_length
                        [0, 0, 0, 1, 0],        # start_seq = 3 , end_seq = seq_length
                        [0, 0, 0, 0, 0]]])      # start_seq = 4 , end_seq = seq_length """ 
    return mask


def get_useful_ones(out, label, mask):
    # get mask, mask the padding and down triangle

    mask = mask.reshape(-1)
    tmp_out = out.reshape(-1, out.shape[-1])
    tmp_label = label.reshape(-1)
    # index select, for gpu speed
    indices = mask.nonzero(as_tuple=False).squeeze(-1).long()
    tmp_out = tmp_out.index_select(0, indices)
    tmp_label = tmp_label.index_select(0, indices)

    return tmp_out, tmp_label