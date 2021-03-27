
import pandas as pd

def _get_text_data(text_data_path):
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
    # bc_dict, sc_dict = _get_text_data("data/SMP-CAIL2020-text-train.csv")
    # print(len(bc_dict.keys()))
    # print(len(sc_dict.keys()))
    a = "中国打到美国"
    b = "打到美"
    res = a.find(b)
    c = a[0:res] + "$" + a[res:res+len(b)] + "$" + a[res+len(b):]
    print(res)
    print(c)