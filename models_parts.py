import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoTokenizer, AutoModel

class Bert(nn.Module):
    def __init__(self, trainable = False, use_rus_version = False):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if use_rus_version == False:
            self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased').to(self.device)
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        else:
            self.bert = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

        if trainable == False:
            for m in self.bert.modules():
                for name, params in m.named_parameters():
                    params.requires_grad = False
        self.target_indx = 0

    def forward(self, input):

        encoding = self.tokenizer.batch_encode_plus(
            input,  # List of input texts
            padding="max_length",
            max_length=512,  # Pad to the maximum sequence length
            truncation=True,  # Truncate to the maximum sequence length if necessary
            return_tensors='pt',  # Return PyTorch tensors
            add_special_tokens=True  # Add special tokens CLS and SEP
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        out = self.bert(input_ids, attention_mask, output_hidden_states=True)
        out = out[0][:,self.target_indx,:]

        return out


class Bert_RUS(nn.Module):
    def __init__(self, trainable):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert = AutoModel.from_pretrained("cointegrated/rubert-tiny").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
        if trainable == False:
            for m in self.bert.modules():
                for name, params in m.named_parameters():
                    params.requires_grad = False
        self.target_indx = 0

    def forward(self, input):

        encoding = self.tokenizer.batch_encode_plus(
            input,  # List of input texts
            padding="max_length",
            max_length=512,  # Pad to the maximum sequence length
            truncation=True,  # Truncate to the maximum sequence length if necessary
            return_tensors='pt',  # Return PyTorch tensors
            add_special_tokens=True  # Add special tokens CLS and SEP
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        out = self.bert(input_ids, attention_mask, output_hidden_states=True)
        out = out[0][:,self.target_indx,:]

        return out

model = Bert(trainable = False, use_rus_version = True)

a = model(['Привет'])

print(a)