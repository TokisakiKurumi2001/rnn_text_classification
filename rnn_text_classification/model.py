import torch
import torch.nn as nn
from rnn_text_classification import tokenizer

class RNNForSeqClassifier(nn.Module):
    def __init__(
        self, num_classes, cell, embed_dim, hidden_dim, num_layers,
        dropout, bidirectional
    ):
        super(RNNForSeqClassifier, self).__init__()
        self.vocab_size = tokenizer.vocab_size
        self.word_embed = nn.Embedding(self.vocab_size, embed_dim)

        # rnn cell
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_dict = {
            'input_size': embed_dim,
            'hidden_size': hidden_dim,
            'num_layers': num_layers,
            'batch_first': True,
            'dropout': dropout,
            'bidirectional': bidirectional
        }
        if cell=="lstm":
            self.cell = nn.LSTM(**self.cell_dict)
        elif cell=="gru":
            self.cell = nn.GRU(**self.cell_dict)
        else:
            self.cell = nn.RNN(**self.cell_dict)

        # projection to classification
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, inputs):
        input_ids = inputs.input_ids
        we_inputs = self.word_embed(input_ids) # (B, L, W)
        output, _ = self.cell(we_inputs) # (B, L, H)
        output = (output[:, -1, :self.hidden_dim] + output[:, -1, self.hidden_dim:])/2
        proj = self.proj(output)
        proj = self.tanh(self.dropout(proj))
        cls_output = self.out(proj)
        return cls_output
