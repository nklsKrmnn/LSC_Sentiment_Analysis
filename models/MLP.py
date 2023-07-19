import transformers
from transformers import BertTokenizer, BertModel, BertConfig
import torch
import torch.nn as nn

class Class_FC(torch.nn.Module):
    def __init__(self):
        super(Class_FC, self).__init__()
        self.l1 = torch.nn.Dropout(0.2)
        self.l2 = torch.nn.Linear(768, 3)

    def forward(self, input, device):

        input = input['input'].to(device, torch.float32)

        output_1 = self.l1(input)
        output = self.l2(output_1)
        return output

class Class_2FC_mse(torch.nn.Module):
    def __init__(self):
        super(Class_2FC_mse, self).__init__()
        self.l1 = torch.nn.Dropout(0.2)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(768, 200)
        self.l4 = torch.nn.Dropout(0.2)
        self.l5 = torch.nn.ReLU()
        self.l6 = torch.nn.Linear(200, 3)
        self.l7 = torch.nn.Dropout(0.2)
        self.l8 = torch.nn.ReLU()
        self.l9 = torch.nn.Linear(3, 1)


    def forward(self, input, device):

        input = input['input'].to(device, torch.float32)

        output_1 = self.l1(input)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output_5 = self.l5(output_4)
        output_6 = self.l6(output_5)
        output_7 = self.l7(output_6)
        output_8 = self.l8(output_7)
        output = self.l9(output_8)

        return output

class Class_2FC_BPP(torch.nn.Module):
    def __init__(self):
        super(Class_2FC_BPP, self).__init__()
        self.l1 = torch.nn.Dropout(0.2)
        self.l2 = torch.nn.Linear(768, 200)
        self.l3 = torch.nn.Dropout(0.2)
        self.l4 = torch.nn.Linear(200, 3)


    def forward(self, input, device):

        input = input['input'].to(device, torch.float32)

        output_1 = self.l1(input)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output = self.l4(output_3)
        return output

class Class_MLP_2RES(torch.nn.Module):
    def __init__(self):
        super(Class_MLP_2RES, self).__init__()
        self.l1 = torch.nn.Dropout(0.2)
        self.l3 = nn.BatchNorm1d(768)
        self.l4 = nn.LayerNorm(768)
        self.l5 = torch.nn.ReLU()
        self.l6 = torch.nn.Linear(768, 100)
        self.l7 = nn.BatchNorm1d(100)
        self.l8 = nn.LayerNorm(100)
        self.l9 = torch.nn.ReLU()
        self.l10 = torch.nn.Linear(100, 768)
        self.l11 = torch.nn.Identity(768, 768)
        self.l12 = torch.nn.Linear(768, 1)

    def forward(self, input, device):

        input = input['input'].to(device, torch.float32)

        output_2 = self.l1(input)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output_5 = self.l5(output_4)
        output_6 = self.l6(output_5)
        output_7 = self.l7(output_6)
        output_8 = self.l8(output_7)
        output_9 = self.l9(output_8)
        output_10 = self.l10(output_9)
        output_11 = self.l11(output_10) + output_2
        output = self.l12(output_11)
        """
        output_7 = self.l7(output_6)
        output_71 = self.l71(output_7)
        output_8 = self.l8(output_71)
        output_81 = self.l81(output_8)
        output_9 = self.l9(output_81)
        output_10 = self.l10(output_9)
        output_11 = self.l11(output_10) + output_71
        output_12 = self.l12(output_11)
        """

        return output
