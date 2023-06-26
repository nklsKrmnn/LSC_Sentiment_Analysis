import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import transformers
from transformers import BertTokenizer, BertModel, BertConfig
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import warnings

warnings.filterwarnings('ignore')
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
device

train_set = pd.read_csv("train.tsv", sep='\t')

train_set_full_sentences = train_set.groupby('SentenceId').first().reset_index()
# %%
data_preprocessed = train_set_full_sentences.join(
    pd.get_dummies(train_set_full_sentences['Sentiment'], dtype=float)).drop('Sentiment', axis=1).iloc[:5][:]

fake_data= [[]]

# Defining some key variables that will be used later on in the training
MAX_LEN = 200
TRAIN_BATCH_SIZE = 30
VALID_BATCH_SIZE = 500
EPOCHS = 1
LEARNING_RATE = 1e-04
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class dataset(Dataset):

    def __init__(self, input, targets, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.input = input
        self.targets = targets
        self.max_len = max_len

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        text = str(self.input[index])
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

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[index].values, dtype=torch.float)
        }


# Creating the dataset and dataloader for the neural network

train_size = 0.8
train_dataset = data_preprocessed.sample(frac=train_size, random_state=200)
test_dataset = data_preprocessed.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(train_set_full_sentences.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset['Phrase'], train_dataset.loc[:, 0:], tokenizer, MAX_LEN)
testing_set = dataset(test_dataset['Phrase'], test_dataset.loc[:, 0:], tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0
               }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)


# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 5)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


model = BERTClass()
model.load_state_dict(torch.load("model.pth"))
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        # if _ % 5000 == 0:
        print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


for epoch in range(EPOCHS):
    train(epoch)


def validation(epoch):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs).argmax(axis=1)
    targets = np.array(targets).argmax(axis=1)
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

torch.save(model.state_dict(), "model.pth")