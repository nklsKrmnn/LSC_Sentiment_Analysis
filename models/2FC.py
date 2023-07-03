import transformers
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn

class Class_2FC(torch.nn.Module):
    def __init__(self):
        super(BERTClass_2FC, self).__init__()
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(768, 200)
        self.l4 = torch.nn.Dropout(0.2)
        self.l5 = torch.nn.Linear(200, 3)

    def forward(self, input, device):

        output_1 = self.l2(input)
        output_2 = self.l3(output_1)
        output_3 = self.l4(output_2)
        output = self.l5(output_3)
        return output