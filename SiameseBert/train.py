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
from model3 import CusArgModel
from utils import MyDataSet, get_acc, FGM

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
seed = 43
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
    log = Logger('training_random_seed.log', level='info')
    fgm = FGM(model)
    for data in tqdm.tqdm(train_iter):
        # convert data to gpu
        bc_input_ids, sc_input_ids, inp, attention_mask, input_type_mask, r_inp, r_attention_mask, r_input_type_mask, label = data
        bc_input_ids = torch.stack(bc_input_ids).permute(1, 0).to(device)
        sc_input_ids = torch.stack(sc_input_ids).permute(1, 0).to(device)
        inp = torch.stack(inp).permute(1, 0).to(device)
        attention_mask = torch.stack(attention_mask).permute(1, 0).to(device)
        input_type_mask = torch.stack(input_type_mask).permute(1, 0).to(device)
        r_inp = torch.stack(r_inp).permute(1, 0).to(device)
        r_attention_mask = torch.stack(r_attention_mask).permute(1, 0).to(device)
        r_input_type_mask = torch.stack(r_input_type_mask).permute(1, 0).to(device)
        label = torch.tensor(label).to(device)

        # begin to model to train
        loss, outputs = model(inp, attention_mask, input_type_mask,
                              r_inp, r_attention_mask, r_input_type_mask, bc_input_ids, sc_input_ids, next_sentence_label=label)
        
        # loss = criterion(outputs, label)
        total_loss += loss.item()
        acc = get_acc(outputs, label)
        total_acc += acc
        log.logger.info("the train acc is: %f" % acc)

        loss.backward()  # 反向传播，得到正常的grad
        # 对抗训练, 在embedding上添加对抗扰动
        fgm.attack()  
        loss_adv, outputs = model(inp, attention_mask, input_type_mask, 
            r_inp, r_attention_mask, r_input_type_mask, bc_input_ids, sc_input_ids, next_sentence_label=label)
        
        # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        loss_adv.backward()

        # 恢复embedding参数  
        fgm.restore()  

        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()
        steps += 1
    return total_loss / steps, total_acc / steps


def valid(model, valid_iter, criterion, device="cuda"):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        steps = 0
        log = Logger('valid_random_seed.log', level='info')
        labels = []
        preds = []
        for data in tqdm.tqdm(valid_iter):
            bc_input_ids, sc_input_ids, inp, attention_mask, input_type_mask, r_inp, r_attention_mask, r_input_type_mask, label = data
            bc_input_ids = torch.stack(bc_input_ids).permute(1, 0).to(device)
            sc_input_ids = torch.stack(sc_input_ids).permute(1, 0).to(device)
            inp = torch.stack(inp).permute(1, 0).to(device)
            attention_mask = torch.stack(attention_mask).permute(1, 0).to(device)
            input_type_mask = torch.stack(input_type_mask).permute(1, 0).to(device)
            r_inp = torch.stack(r_inp).permute(1, 0).to(device)
            r_attention_mask = torch.stack(r_attention_mask).permute(1, 0).to(device)
            r_input_type_mask = torch.stack(r_input_type_mask).permute(1, 0).to(device)
            label = torch.tensor(label).to(device)

            loss, outputs = model(inp, attention_mask, input_type_mask,
                                  r_inp, r_attention_mask, r_input_type_mask, bc_input_ids, sc_input_ids, next_sentence_label=label)

            labels.append(label.item())
            preds.append(outputs[0][1].item())

            total_loss += loss.item()
            acc = get_acc(outputs, label)
            total_acc += acc
            log.logger.info("the valid acc is: %f" % acc)

            steps += 1
        answer_list = []
        label_answer_list = []
        for i in range(0, len(preds), 5):
            logits = preds[i:i + 5]
            answer = int(torch.argmax(torch.tensor(logits)))
            answer_list.append(answer + 1)

            logits = labels[i:i + 5]
            answer = int(torch.argmax(torch.tensor(logits)))
            label_answer_list.append(answer + 1)

        answer_list = np.array(answer_list)
        label_answer_list = np.array(label_answer_list)
        res = answer_list == label_answer_list
        cnt = 0
        for x in res:
            if x == True:
                cnt += 1
        log.logger.info("the evalute acc is: %f" % (cnt / len(res)))

        return total_loss / steps, total_acc / steps, cnt / len(res)


if __name__ == '__main__':
    pretrain_path = "pretrain_model/ERNIE_1.0_max-len-512-pytorch"
    save_path = "models/best.pt"
    batch_size = 1
    lr = 5e-5
    epoch = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    if torch.cuda.device_count() > 1:
        print('the gpu nums: ', torch.cuda.device_count())
    model = CusArgModel(pretrain_path)
    model.to(device)
    # model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # multi-GPU
    # model.load_state_dict(torch.load(save_path))

    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    opimizer = AdamW(model.parameters(), lr=lr)

    # 训练数据集
    train_dataset = MyDataSet(vocab_file=pretrain_path, text_path="data/SMP-CAIL2020-text-train.csv", 
    file_path="data/train.csv", train_type="train")
    
    # 训练迭代器
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 验证数据集
    valid_dataset = MyDataSet(vocab_file=pretrain_path, text_path="data/SMP-CAIL2020-text-train.csv", 
    file_path="data/valid.csv", train_type="valid")

    # 验证迭代器
    valid_iter = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    valid_best_loss = float("inf")
    valid_best_acc = 0.0
    log = Logger('trainRecord_random_seed.log', level='info')
    for e in range(epoch):
        print("epoch:{}".format(e + 1))
        log.logger.info("epoch:{}".format(e + 1))
        
        # 验证数据
        train_loss, train_acc = train(model, train_iter, opimizer, criterion, device)
        print("train_acc:{} train_loss:{}".format(train_acc, train_loss))
        
        # 记录训练日志
        log.logger.info("train_acc:{} train_loss:{}".format(train_acc, train_loss))

        # 验证集损失函数
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
