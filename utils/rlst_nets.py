import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self,
                 src_vocab,
                 trg_vocab,
                 rnn_hid_dim,
                 dropout,
                 rnn_num_layers):
        super().__init__()

        src_emb_dim = src_vocab.vectors.size()[1]
        trg_emb_dim = trg_vocab.vectors.size()[1]
        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_num_layers = rnn_num_layers
        self.src_embedding = nn.Embedding(len(src_vocab), src_emb_dim).from_pretrained(src_vocab.vectors, freeze=True)
        self.trg_embedding = nn.Embedding(len(trg_vocab), trg_emb_dim).from_pretrained(trg_vocab.vectors, freeze=True)
        self.rnn = nn.GRU(src_emb_dim + trg_emb_dim, rnn_hid_dim, num_layers=rnn_num_layers, bidirectional=False, dropout=dropout)
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 3)
        nn.init.constant_(self.output.bias[-3:], -10.0)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.src_embedding(src)
        trg_embedded = self.trg_embedding(previous_output)
        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        outputs = self.output(rnn_output)
        return outputs, rnn_state


class Net1(nn.Module):
    def __init__(self, src_vocab, trg_vocab, rnn_hid_dim,
                 dropout, rnn_num_layers, linear_dim):

        super().__init__()
        src_emb_dim = src_vocab.vectors.size()[1]
        trg_emb_dim = trg_vocab.vectors.size()[1]
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_num_layers = rnn_num_layers
        self.src_embedding = nn.Embedding(len(src_vocab), src_emb_dim).from_pretrained(src_vocab.vectors, freeze=True)
        self.trg_embedding = nn.Embedding(len(trg_vocab), trg_emb_dim).from_pretrained(trg_vocab.vectors, freeze=True)
        self.rnn = nn.GRU(src_emb_dim + trg_emb_dim, rnn_hid_dim, num_layers=rnn_num_layers, bidirectional=False, dropout=dropout)
        self.linear = nn.Linear(rnn_hid_dim, linear_dim)
        self.output = nn.Linear(linear_dim, len(trg_vocab) + 3)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.src_embedding(src)
        trg_embedded = self.trg_embedding(previous_output)
        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        linear_output = self.relu(self.linear(self.dropout(rnn_output)))
        outputs = self.output(linear_output)
        return outputs, rnn_state


class Net2(nn.Module):
    def __init__(self,
                 src_vocab,
                 trg_vocab,
                 rnn_hid_dim,
                 dropout,
                 rnn_num_layers,
                 linear_dim):
        super().__init__()

        src_emb_dim = src_vocab.vectors.size()[1]
        trg_emb_dim = trg_vocab.vectors.size()[1]
        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_num_layers = rnn_num_layers
        self.src_embedding = nn.Embedding(len(src_vocab), src_emb_dim).from_pretrained(src_vocab.vectors, freeze=True)
        self.trg_embedding = nn.Embedding(len(trg_vocab), trg_emb_dim).from_pretrained(trg_vocab.vectors, freeze=True)
        self.linear = nn.Linear(src_emb_dim + trg_emb_dim, linear_dim)
        self.activation = nn.ReLU()
        self.rnn = nn.GRU(linear_dim, rnn_hid_dim, num_layers=rnn_num_layers, bidirectional=False, dropout=dropout)
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.src_embedding(src)
        trg_embedded = self.trg_embedding(previous_output)
        rnn_input = self.dropout(self.activation(self.linear(torch.cat((src_embedded, trg_embedded), dim=2))))
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        outputs = self.output(rnn_output)
        return outputs, rnn_state
