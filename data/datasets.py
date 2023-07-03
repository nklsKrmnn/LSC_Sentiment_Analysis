import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class dataset(Dataset):

    def __init__(self, input, targets,
                 tokenize_bert: bool, onehot: bool = True, second_layer: bool = False,
                 tokenizer=None, max_len=None):
        self.input = input
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.onehot = onehot
        self.tokenize = tokenize_bert
        self.second_layer = second_layer

        if self.onehot:
            # One hot encoding
            print("onehot encoding")
            targets = pd.DataFrame(self.targets, columns=['Sentiment'])
            targets = pd.get_dummies(targets['Sentiment'], dtype=float)#.drop('Sentiment', axis=1)
            self.targets = targets.values.tolist()



    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input = self.input[index]
        target = self.targets[index]

        #TODO: weiters preprocessing

        # if self.capitalize:

        if self.tokenize:
            # Error Check
            if self.tokenizer is None:
                raise Exception("[Dataset]: No proper 'tokenizer' found!")
            if self.max_len is None:
                raise Exception("[Dataset]: No proper 'max_len' found!")

            text = str(input)
            text = " ".join(text.split())

            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_token_type_ids=True,
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            token_type_ids = inputs["token_type_ids"]

            item = {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(target, dtype=torch.float)
            }
        elif self.second_layer:
            item = {
                'input': torch.tensor(self.input[index], dtype=torch.float32),
                'targets': torch.tensor(self.targets[index], dtype=torch.float)
            }
        else:
            item = {
                'input': torch.tensor(self.input[index], dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.float)
            }
        return item
