from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from torch.utils.data import Dataset

from torch.utils.data import TensorDataset
from transformers import BertTokenizer
from tqdm import tqdm

import time

def get_acc(y_preds, y_labels):
    total = y_preds.shape[0]
    pred_label = y_preds.argmax(dim=1)
    num_correct = (pred_label == y_labels).sum().item()
    # print(y_labels.size())
    # print(pred_label.size())
    return num_correct / total


class FGM():
    """
    参考:  https://zhuanlan.zhihu.com/p/91269728
    """
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class CusDataSet(Dataset):
    def __init__(self, data_path, tokenizer, test=False):
        super(CusDataSet, self).__init__()
        self.test = test
        self.dataset = []
        self.dataset_input_type_mask = []
        self.dataset_mask = []
        self.r_dataset = []
        self.r_dataset_input_type_mask = []
        self.r_dataset_mask = []
        self.label_list = []

        max_len = 512

        with open(data_path, encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                sc = line[1]
                bc = line[3]
                inputs = tokenizer.encode_plus(text=sc, text_pair=bc, max_length=max_len, pad_to_max_length=True)
                input_ids = inputs["input_ids"]
                token_type_ids = inputs["token_type_ids"]
                attention_mask = inputs["attention_mask"]

                r_inputs = tokenizer.encode_plus(text=bc, text_pair=sc, max_length=max_len, pad_to_max_length=True)
                r_input_ids = r_inputs["input_ids"]
                r_token_type_ids = r_inputs["token_type_ids"]
                r_attention_mask = r_inputs["attention_mask"]


                self.dataset.append(input_ids)
                self.dataset_input_type_mask.append(token_type_ids)
                self.dataset_mask.append(attention_mask)

                self.r_dataset.append(r_input_ids)
                self.r_dataset_input_type_mask.append(r_token_type_ids)
                self.r_dataset_mask.append(r_attention_mask)
                if test == False:
                    self.label_list.append(int(line[-1]))
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # print(item)
        # print(len(self.dataset))
        if self.test is True:
            return self.dataset[item],  self.dataset_input_type_mask[item], \
                   self.dataset_mask[item], self.r_dataset[item], \
                   self.r_dataset_input_type_mask, self.r_dataset_mask
        return self.dataset[item],  self.dataset_input_type_mask[item], \
                   self.dataset_mask[item], self.r_dataset[item], \
                   self.r_dataset_input_type_mask, self.r_dataset_mask, \
               self.label_list[item]



class MyDataSet(Dataset):

    def __init__(self, vocab_file='', file_path='', text_path='',train_type="train", max_seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(vocab_file)
        self.max_seq_len = max_seq_len
        self.train_type = train_type

        bc_dict, sc_dict = self._get_text_data(text_path)
        idx_list, sc_list, bc_list, label_list = self._load_file(file_path, train_type)
        dataset = self._convert_sentence_pair_to_bert_dataset(
            sc_list, bc_list, bc_dict, sc_dict, idx_list, label_list=label_list)
        if train_type == "train" or train_type == "valid":
            self.all_bc_input_ids, self.all_sc_input_ids, self.all_input_ids, self.all_attention_mask, self.all_token_type_ids, \
            self.r_all_input_ids, self.r_all_attention_mask, self.r_all_token_type_ids,\
            self.all_label_ids = dataset

        elif train_type=="test":
            self.all_bc_input_ids, self.all_sc_input_ids, self.all_input_ids, self.all_attention_mask, self.all_token_type_ids, \
            self.r_all_input_ids, self.r_all_attention_mask, self.r_all_token_type_ids = dataset
        else:
            exit(4)

    def __len__(self):
        return len(self.all_input_ids)

    def __getitem__(self, item):
        if self.train_type=="train" or self.train_type=="valid":
            return self.all_bc_input_ids[item], self.all_sc_input_ids[item], self.all_input_ids[item], self.all_attention_mask[item], \
                   self.all_token_type_ids[item], \
                   self.r_all_input_ids[item], self.r_all_attention_mask[item], self.r_all_token_type_ids[item], \
                   self.all_label_ids[item]
        else:
            return self.all_bc_input_ids[item], self.all_sc_input_ids[item], self.all_input_ids[item], self.all_attention_mask[item], \
                   self.all_token_type_ids[item],\
                   self.r_all_input_ids[item], self.r_all_attention_mask[item], self.r_all_token_type_ids[item]

    def _load_text_file(self):
        pass

    def _load_file(self, filename, train_type="train"):

        data_frame = pd.read_csv(filename)
        idx_list, sc_list, bc_list, label_list = [], [], [], []
        for row in data_frame.itertuples(index=False):
            candidates = row[3:8]
            idx = int(row[1])
            if train_type=="train" or train_type=="valid":
                answer = int(row[-1])

            # sc_tokens = self.tokenizer.tokenize(row[2])
            sc_tokens = row[2]
            for i, _ in enumerate(candidates):
                # bc_tokens = self.tokenizer.tokenize(candidates[i])
                bc_tokens = candidates[i]
                if train_type=="train":
                    if i + 1 == answer:
                        # Copy positive sample 4 times
                        for _ in range(len(candidates) - 1):
                            idx_list.append(idx)
                            sc_list.append(sc_tokens)
                            bc_list.append(bc_tokens)
                            label_list.append(1)
                    else:
                        idx_list.append(idx)
                        sc_list.append(sc_tokens)
                        bc_list.append(bc_tokens)
                        label_list.append(0)
                elif train_type=="valid":
                    idx_list.append(idx)
                    if i + 1 == answer:
                        sc_list.append(sc_tokens)
                        bc_list.append(bc_tokens)
                        label_list.append(1)
                    else:
                        sc_list.append(sc_tokens)
                        bc_list.append(bc_tokens)
                        label_list.append(0)
                else:  # test
                    idx_list.append(idx)
                    sc_list.append(sc_tokens)
                    bc_list.append(bc_tokens)
        return idx_list, sc_list, bc_list, label_list

    def _convert_sentence_pair_to_bert_dataset(
            self, s1_list, s2_list, bc_dict, sc_dict, idx_list, label_list=None):
                # sc_list, bc_list, bc_dict, sc_dict, idx_list, label_list
        all_input_ids, all_attention_mask, all_token_type_ids = [], [], []
        r_all_input_ids, r_all_attention_mask, r_all_token_type_ids = [], [], []
        all_bc_input_ids, all_sc_input_ids = [], []
        # print("idx_list:  ", idx_list)
        for i, _ in tqdm(enumerate(s1_list), ncols=80):
            idx = idx_list[i]
            # print("idx: ", idx, "\n")
            start = time.time()
            sc_sentence, bc_sentence = s1_list[i], s2_list[i]
            sc_text, bc_text = sc_dict[idx], bc_dict[idx]
            res = sc_text.find(sc_sentence)
            sc_text = sc_text[0:res] + "$" + sc_text[res:res + len(sc_sentence)] + "$" + sc_text[res + len(sc_sentence):]
            res = bc_text.find(bc_sentence)
            bc_text = bc_text[0:res] + "$" + bc_text[res:res + len(bc_sentence)] + "$" + bc_text[res + len(bc_sentence):]
            # print("sc_sentence: ", sc_sentence)
            # print("bc_sentence: ", bc_sentence)
            # print("sc: ", sc_text)
            # print("bc: ", bc_text)
            # exit(3)
            bc_input_ids = self.tokenizer.encode_plus(text=bc_text, add_special_tokens=True, max_length=512, pad_to_max_length=True)["input_ids"]
            sc_input_ids = self.tokenizer.encode_plus(text=sc_text, add_special_tokens=True, max_length=512, pad_to_max_length=True)["input_ids"]
            all_bc_input_ids.append(bc_input_ids)
            all_sc_input_ids.append(sc_input_ids)

            inputs = self.tokenizer.encode_plus(text=s1_list[i], text_pair=s2_list[i], max_length=512, pad_to_max_length=True)
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_token_type_ids.append(token_type_ids)

            r_inputs = self.tokenizer.encode_plus(text=s2_list[i], text_pair=s1_list[i], max_length=512, pad_to_max_length=True)
            r_input_ids = r_inputs["input_ids"]
            r_token_type_ids = r_inputs["token_type_ids"]
            r_attention_mask = r_inputs["attention_mask"]
            r_all_input_ids.append(r_input_ids)
            r_all_attention_mask.append(r_attention_mask)
            r_all_token_type_ids.append(r_token_type_ids)

        if label_list:  # train, valid
            all_label_ids = torch.tensor(label_list, dtype=torch.long)
            return all_bc_input_ids, all_sc_input_ids, all_input_ids, all_attention_mask, all_token_type_ids, \
                   r_all_input_ids, r_all_attention_mask, r_all_token_type_ids,\
                   all_label_ids
        # test
        return all_bc_input_ids, all_sc_input_ids, all_input_ids, all_attention_mask, all_token_type_ids, \
                   r_all_input_ids, r_all_attention_mask, r_all_token_type_ids,

    def _get_text_data(self, text_data_path):
        data_frame = pd.read_csv(text_data_path)
        # print(data_frame.head(3))

        sc_dict = {}
        bc_dict = {}

        for row in data_frame.itertuples(index=False):
            # print(row)
            text_id = int(row[1])
            position = row[2]
            sentence = row[3].strip()
            # break

            if position == "bc":
                if text_id in bc_dict.keys():
                    bc_dict[text_id] += sentence
                else:
                    bc_dict[text_id] = sentence
            elif position == "sc":
                if text_id in sc_dict.keys():
                    sc_dict[text_id] += sentence
                else:
                    sc_dict[text_id] = sentence
            else:
                print("position error")
                exit(1)

        return bc_dict, sc_dict







if __name__ == '__main__':
    from transformers import BertTokenizer
    import torch
    path ="F:/bert/ERNIE_1.0_max-len-512-pytorch"
    tokenizer = BertTokenizer.from_pretrained(path)
    cus_dataset = MyDataSet("data/valid.csv", tokenizer)
    valid_iter = DataLoader(cus_dataset, batch_size=1, shuffle=True)

    # num =3
    # for data in valid_iter:
    #     print(len(data))
    #     dataset,dataset_input_type_mask, dataset_mask = data
    #     # print(torch.stack(sc).permute(1,0))
    #     # print(torch.stack(bc))
    #     # print(torch.tensor(label))
    #     # exit(3)
    #     num -= 1
    #     if num == 0:
    #         break


