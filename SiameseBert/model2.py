from transformers import (BertTokenizer, BertConfig,  BertForSequenceClassification, AutoModel, BertForNextSentencePrediction,
                          BertModel, get_linear_schedule_with_warmup,
                          AdamW)
                          
import torch
import torch.nn as nn
BertLayerNorm = torch.nn.LayerNorm


class CusBertForNextSentencePrediction(nn.Module):
    def __init__(self, pre_train_path):
        super(CusBertForNextSentencePrediction, self).__init__()
        config = BertConfig.from_pretrained(pre_train_path)
        self.bert_model = BertForNextSentencePrediction.from_pretrained(pre_train_path, config=config)


    def forward(self, inp, attention_mask, input_type_mask, label=None):
        if label is not None:
            outputs = self.bert_model(inp, attention_mask, input_type_mask, next_sentence_label=label)
            return outputs[0], outputs[1] # output[0] is loss, outputs[1] is score
        else:
            outputs = self.bert_model(inp, attention_mask, input_type_mask)
            return outputs[0] # output[0] is score


if __name__ == '__main__':
    pretrain_path = "pretrain_model/ERNIE_1.0_max-len-512-pytorch"
    model = CusBertForNextSentencePrediction(pretrain_path)
    # ArgModel(pretrain_path)
    print(model)