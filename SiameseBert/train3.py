import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer
import os
from costumLog import Logger
from logging import log
import tqdm
import random
import numpy as np

from model2 import CusBertForNextSentencePrediction
from utils import MyDataSet ,get_acc

os.environ['CUDA_VIDIBLE_DEVICES'] = '0'
seed = 1024
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def train(model, train_iter, optimizer, criterion, device="cuda"):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    log = Logger('training.log', level='info')
    for data in tqdm.tqdm(train_iter):
        inp, attention_mask, input_type_mask, label = data
        inp = torch.stack(inp).permute(1, 0).to(device)
        attention_mask = torch.stack(attention_mask).permute(1, 0).to(device)
        input_type_mask = torch.stack(input_type_mask).permute(1, 0).to(device)

        label = torch.tensor(label).to(device)

        loss, outputs = model(inp, attention_mask, input_type_mask, label)
        # loss = criterion(outputs, label)
        total_loss += loss.item()
        acc = get_acc(outputs, label)
        total_acc += acc
        log.logger.info("the train acc is: %f" % acc)

        opimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps += 1
    return total_loss/steps, total_acc/steps


def valid(model, valid_iter, criterion, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    # log = Logger('valid.log', level='info')
    labels = []
    preds = []
    for data in tqdm.tqdm(valid_iter):
        inp, attention_mask, input_type_mask, label = data
        inp = torch.stack(inp).permute(1, 0).to(device)
        attention_mask = torch.stack(attention_mask).permute(1, 0).to(device)
        input_type_mask = torch.stack(input_type_mask).permute(1, 0).to(device)
        label = torch.tensor(label).to(device)

        loss, outputs = model(inp, attention_mask, input_type_mask, label)

        labels.append(label.item())
        preds.append(outputs[0][1].item())

        total_loss += loss.item()
        acc = get_acc(outputs, label)
        total_acc += acc
        # log.logger.info("the valid acc is: %f" % acc)

        steps += 1
    answer_list = []
    label_answer_list = []
    for i in range(0, len(preds), 5):
        logits = preds[i:i + 5]
        answer1 = int(torch.argmax(torch.tensor(logits)))
        answer_list.append(answer1 + 1)

        logits = labels[i:i + 5]
        answer2 = int(torch.argmax(torch.tensor(logits)))
        print(answer1+1, ' ', answer2+1)
        label_answer_list.append(answer2 + 1)
    print(answer_list)
    print(label_answer_list)
    answer_list = np.array(answer_list)
    label_answer_list = np.array(label_answer_list)
    res = answer_list == label_answer_list
    cnt = 0
    for x in res:
        if x == True:
            cnt += 1
    log.logger.info("the evalute acc is: %f" % (cnt/len(res)))

    return total_loss / steps, total_acc / steps, cnt/len(res)




if __name__ == '__main__':
    pretrain_path = "pretrain_model/ERNIE_1.0_max-len-512-pytorch"
    save_path = "models/best.pt"
    batch_size = 5
    lr = 1e-6
    epoch = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    if torch.cuda.device_count() > 1:
        print('the gpu nums: ', torch.cuda.device_count())
    model = CusBertForNextSentencePrediction(pretrain_path)
    model.to(device)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # multi-GPU
    model.load_state_dict(torch.load(save_path))

    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    opimizer = AdamW(model.parameters(), lr=lr)

    # train_dataset = MyDataSet(vocab_file=pretrain_path, file_path="data/train.csv", train_type="train")
    # train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = MyDataSet(vocab_file=pretrain_path, file_path="data/valid.csv", train_type="valid")
    valid_iter = DataLoader(valid_dataset, batch_size=1, shuffle=False)


    criterion = nn.CrossEntropyLoss()
    valid_best_loss = float("inf")
    valid_best_acc = 0.0
    log = Logger('trainRecord.log', level='info')
    for e in range(epoch):
        print("epoch:{}".format(e+1))
        # log.logger.info("epoch:{}".format(e+1))
        # train_loss, train_acc = train(model, train_iter, opimizer, criterion, device)
        # print("train_acc:{} train_loss:{}".format(train_acc, train_loss))
        # log.logger.info("train_acc:{} train_loss:{}".format(train_acc, train_loss))
        valid_loss, valid_acc, evaluate_acc = valid(model, valid_iter, criterion, device)
        print("valid_acc:{} valid_loss:{} evaluate_acc:{}".format(valid_acc, valid_loss, evaluate_acc))
        log.logger.info("valid_acc:{} valid_loss:{} evalute_acc:{}".format(valid_acc, valid_loss, evaluate_acc))
        if evaluate_acc > valid_best_acc:
            valid_best_acc = evaluate_acc
            valid_best_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            save_path2 = save_path + "2"
            torch.save(model, save_path2)
            print("save best model, evaluate_acc:{}".format(evaluate_acc))
            log.logger.info("the best evaluate_acc model is %f " % evaluate_acc)

    print("evaluate_acc:{}".format(valid_best_acc))
    print("finished!")

