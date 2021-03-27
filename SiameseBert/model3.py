
import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel, BertConfig

class CusBertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        return pooled_output


class CusArgModel(nn.Module):
    def __init__(self, pre_train_path):
        super(CusArgModel, self).__init__()
        config = BertConfig.from_pretrained(pre_train_path)
        self.bert_model = CusBertForNextSentencePrediction.from_pretrained(pre_train_path,config=config)

        self.bert_text = BertModel.from_pretrained(pre_train_path,config=config)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.Tanh()
        self.cls = nn.Linear(768*3, 2)

    def forward(self, inp, attention_mask, input_type_mask, r_inp, r_attention_mask,
                r_input_type_mask, bc, sc, next_sentence_label=None):
        pooled_output = self.bert_model(inp, attention_mask, input_type_mask)
        r_pooled_output = self.bert_model(r_inp, r_attention_mask, r_input_type_mask)
        ## 新添加的
        sc_output = self.bert_text(bc)[0] # (bs, max_len, hidden state)
        bc_output = self.bert_text(sc)[0] # (bs, max_len, hidden state)
        sc_output = sc_output[:,0,:] # (bs, hidden state)
        bc_output = bc_output[:,0,:] # (bs, hidden state)

        # print(sc_output.size())
        # print(pooled_output.size())

        #cat_pooled_output = torch.cat((pooled_output, r_pooled_output, sc_output-bc_output, sc_output+bc_output, torch.mul(sc_output, bc_output)), dim=-1)
        cat_pooled_output = torch.cat((pooled_output, r_pooled_output, sc_output-bc_output), dim=-1)
        cat_pooled_output = self.dropout(cat_pooled_output)
        cat_pooled_output = self.relu(cat_pooled_output)

        # out = self.cls(cat_pooled_output)
        seq_relationship_score = self.cls(cat_pooled_output)

        outputs = (seq_relationship_score,)
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss)

