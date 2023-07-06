import transformers
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 3)

    def forward(self, input, device):
        ids = input['ids'].to(device, dtype=torch.long)
        mask = input['mask'].to(device, dtype=torch.long)
        token_type_ids = input['token_type_ids'].to(device, dtype=torch.long)

        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class BERTClass_2FC(torch.nn.Module):
    def __init__(self):
        super(BERTClass_2FC, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.LayerNorm(768)
        self.l4 = torch.nn.Linear(768, 200)
        self.l5 = torch.nn.Dropout(0.3)
        self.l6 = torch.nn.Linear(200, 3)

    def forward(self, input, device):
        ids = input['ids'].to(device, dtype=torch.long)
        mask = input['mask'].to(device, dtype=torch.long)
        token_type_ids = input['token_type_ids'].to(device, dtype=torch.long)

        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output_5 = self.l5(output_4)
        output = self.l6(output_5)
        return output

class BERTClass_res(torch.nn.Module):
    def __init__(self):
        super(BERTClass_2FC, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.9)
        self.l3 = nn.BatchNorm1d(768)
        self.l4 = torch.nn.ReLU()
        self.l5 = torch.nn.Linear(768, 768)
        self.l6 = torch.nn.Identity(768, 768)
        self.l7 = torch.nn.ReLU()
        self.l8 = torch.nn.Linear(768, 3)

    def forward(self, input, device):
        ids = input['ids'].to(device, dtype=torch.long)
        mask = input['mask'].to(device, dtype=torch.long)
        token_type_ids = input['token_type_ids'].to(device, dtype=torch.long)

        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output_5 = self.l5(output_4)
        output_6 = self.l6(output_5) + output_2
        output_7 = self.l7(output_6)
        output = self.l8(output_7)
        return output