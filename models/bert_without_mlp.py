import transformers
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn

class BERTClass_without_mlp(torch.nn.Module):
    def __init__(self):
        super(BERTClass_without_mlp, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)

    def forward(self, input, device):
        ids = input['ids'].to(device, dtype=torch.long)
        mask = input['mask'].to(device, dtype=torch.long)
        token_type_ids = input['token_type_ids'].to(device, dtype=torch.long)

        return self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)