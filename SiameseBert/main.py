import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertConfig
import tqdm
from torch.utils.data import DataLoader

from utils import MyDataSet
from model2 import CusBertForNextSentencePrediction
from model3 import CusArgModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction(model, valid_iter, device="cuda"):
    model.eval()
    with torch.no_grad():
        # log = Logger('valid.log', level='info')
        preds = []
        for data in tqdm.tqdm(valid_iter):
            bc_input_ids, sc_input_ids, inp, attention_mask, input_type_mask, r_inp, r_attention_mask, r_input_type_mask = data
            bc_input_ids = torch.stack(bc_input_ids).permute(1, 0).to(device)
            sc_input_ids = torch.stack(sc_input_ids).permute(1, 0).to(device)
            inp = torch.stack(inp).permute(1, 0).to(device)
            attention_mask = torch.stack(attention_mask).permute(1, 0).to(device)
            input_type_mask = torch.stack(input_type_mask).permute(1, 0).to(device)
            r_inp = torch.stack(r_inp).permute(1, 0).to(device)
            r_attention_mask = torch.stack(r_attention_mask).permute(1, 0).to(device)
            r_input_type_mask = torch.stack(r_input_type_mask).permute(1, 0).to(device)

            outputs = model(inp, attention_mask, input_type_mask,
                                  r_inp, r_attention_mask, r_input_type_mask, bc_input_ids, sc_input_ids)

            # print("outputs: ", outputs)

            preds.append(outputs[0][0][1].item())

        answer_list = []
        for i in range(0, len(preds), 5):
            logits = preds[i:i + 5]
            answer1 = int(torch.argmax(torch.tensor(logits)))
            answer_list.append(answer1 + 1)
        print(answer_list)
        answer_list = np.array(answer_list)

    return answer_list


if __name__ == '__main__':
    pretrain_path = "pretrain_model/ERNIE_1.0_max-len-512-pytorch"
    best_model_path = 'models/best.pt'

    model = CusArgModel(pretrain_path)
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)

    # text_path = "/home/wl/Desktop/lbwj_train/SMP-CAIL2020-Argmine-train/SMP-CAIL2020-text-train.csv"
    # arg_path = "data/valid.csv"
    # output_file = "res.csv"

    text_path = "/input/SMP-CAIL2020-text-test1.csv"
    arg_path = "/input/SMP-CAIL2020-test1.csv"
    output_file = "/output/result1.csv"


    valid_dataset = MyDataSet(vocab_file=pretrain_path, file_path=arg_path, text_path=text_path, train_type="test")
    valid_iter = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    result = prediction(model, valid_iter, device)
    data = pd.read_csv(arg_path)
    all_idxs = data["id"]

    ans = pd.DataFrame(columns=['id', 'answer'])
    for i,res in enumerate(result):
        idx = int(all_idxs[i])
        print(idx)
        ans.loc[idx, 'id'] = idx
        ans.loc[idx, 'answer'] = res

    ans['id'] = ans['id'].astype('int')
    ans['answer'] = ans['answer'].astype('int')
    ans.to_csv(output_file, encoding='utf-8', index=False)

