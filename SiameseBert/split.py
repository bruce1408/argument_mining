import pandas as pd


def get_text_data(text_path):
    data_frame = pd.read_csv(text_path)
    # data_frame.head(1)
    cnt = 0
    text_id_bc_dict, text_id_sc_dict = {}, {}
    for row in data_frame.itertuples(index=False):
        sentence_id = row[0]
        text_id = row[1]
        position = row[2]
        sentence = row[3]
        if len(sentence) >=510:
            cnt += 1
        if position == "bc" and text_id not in text_id_bc_dict.keys():
            text_id_bc_dict[text_id] = sentence
        elif position == "bc":
            text_id_bc_dict[text_id] += sentence
        elif position == "sc" and text_id not in text_id_sc_dict.keys():
            text_id_sc_dict[text_id] = sentence
        elif position == "sc":
            text_id_sc_dict[text_id] += sentence
    print("longer than 510: ", cnt)
    return text_id_bc_dict, text_id_sc_dict

def get_arg(arg_path):
    # data_combined = []
    data_frame = pd.read_csv(arg_path)
    sc_list, bc_list, label_list, text_id_list = [], [], [] , []
    for row in data_frame.itertuples(index=False):
        candidates = row[3:8]
        answer = int(row[-1])
        text_id = row[1]
        sc = row[2]
        for idx, _ in enumerate(candidates):
            if idx + 1 == answer:
                #
                # copy操作
                sc_list.append(sc)
                bc_list.append(candidates[idx])
                label_list.append(1)
                text_id_list.append(text_id)

                sc_list.append(sc)
                bc_list.append(candidates[idx])
                label_list.append(1)
                text_id_list.append(text_id)

                sc_list.append(sc)
                bc_list.append(candidates[idx])
                label_list.append(1)
                text_id_list.append(text_id)

                sc_list.append(sc)
                bc_list.append(candidates[idx])
                label_list.append(1)
                text_id_list.append(text_id)

                sc_list.append(sc)
                bc_list.append(candidates[idx])
                label_list.append(1)
                text_id_list.append(text_id)
            else:
                sc_list.append(sc)
                bc_list.append(candidates[idx])
                label_list.append(0)
                text_id_list.append(text_id)
    return sc_list, bc_list, label_list, text_id_list


def combine_data(text_id_bc_dict, text_id_sc_dict, sc_list, bc_list, label_list, text_id_list):
    data_list = []
    for idx, text_id in enumerate(text_id_list):
        if text_id not in text_id_bc_dict:
            continue
        data = [text_id_sc_dict[text_id], sc_list[idx] ,text_id_bc_dict[text_id], bc_list[idx],
                label_list[idx]]
        data_list.append(data)
    return data_list

def split(text_path, arg_path):
    # 读取text_data
    text_id_bc_dict, text_id_sc_dict = get_text_data(text_path)
    print(len(text_id_sc_dict.keys()), " ", len(text_id_bc_dict.keys()))
    text_id_sc_dict = sorted(text_id_sc_dict.items(), key=lambda item:item[0])
    text_id_bc_dict = sorted(text_id_bc_dict.items(), key=lambda item: item[0])

    # 切分训练集和验证集
    train_text_id_bc_dict = dict(text_id_bc_dict[:-35])
    valid_text_id_bc_dict = dict(text_id_bc_dict[-35:])
    train_text_id_sc_dict = dict(text_id_sc_dict[:-35])
    valid_text_id_sc_dict = dict(text_id_sc_dict[-35:])

    # 读取 data
    sc_list, bc_list, label_list, text_id_list = get_arg(arg_path)

    # 组合数据
    train_data = combine_data(train_text_id_bc_dict, train_text_id_sc_dict,
                              sc_list, bc_list, label_list, text_id_list)
    valid_data = combine_data(valid_text_id_bc_dict, valid_text_id_sc_dict,
                              sc_list, bc_list, label_list, text_id_list)
    print("train nums:{} valid_nums:{}".format(len(train_data), len(valid_data)))

   # 写入文件
    with open("data/train.txt", encoding="utf-8", mode="w") as f:
        for line in train_data:
            f.write(line[0] + "\t" + line[1] + "\t" + line[2] + "\t" + line[3] + "\t" + str(line[4]) + "\n")

    with open("data/valid.txt", encoding="utf-8", mode="w") as f:
        for line in valid_data:
            f.write(line[0] + "\t" + line[1] + "\t" + line[2] + "\t" + line[3] + "\t" + str(line[4]) + "\n")

    return train_data, valid_data


if __name__ == '__main__':
    text_path = "/home/wl/Desktop/lbwj_train/SMP-CAIL2020-Argmine-train/SMP-CAIL2020-text-train.csv"
    arg_path = "/home/wl/Desktop/lbwj_train/SMP-CAIL2020-Argmine-train/SMP-CAIL2020-train.csv"
    split(text_path, arg_path)
