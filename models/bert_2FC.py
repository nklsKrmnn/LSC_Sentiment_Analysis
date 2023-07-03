import transformers
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn

class BERTClass_2FC(torch.nn.Module):
    def __init__(self):
        super(BERTClass_2FC, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(768, 200)
        self.l4 = torch.nn.Dropout(0.2)
        self.l5 = torch.nn.Linear(200, 3)

    def forward(self, input, device):
        ids = input['ids'].to(device, dtype=torch.long)
        mask = input['mask'].to(device, dtype=torch.long)
        token_type_ids = input['token_type_ids'].to(device, dtype=torch.long)

        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output = self.l5(output_4)
        return output