import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self,
                 src_vocab,
                 trg_vocab,
                 use_pretrained_embeddings,
                 rnn_hid_dim,
                 rnn_dropout,
                 rnn_num_layers,
                 src_embed_dim=256,
                 trg_embed_dim=256,
                 embedding_dropout=0.0):
        super().__init__()

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_num_layers = rnn_num_layers

        self.src_embedding = nn.Embedding(len(src_vocab), src_embed_dim)
        self.trg_embedding = nn.Embedding(len(trg_vocab), trg_embed_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        if use_pretrained_embeddings:
            src_embed_dim = src_vocab.vectors.size()[1]
            trg_embed_dim = trg_vocab.vectors.size()[1]
            self.src_embedding = nn.Embedding(len(src_vocab), src_embed_dim).from_pretrained(src_vocab.vectors, freeze=True)
            self.trg_embedding = nn.Embedding(len(trg_vocab), trg_embed_dim).from_pretrained(trg_vocab.vectors, freeze=True)

        self.rnn = nn.GRU(src_embed_dim + trg_embed_dim, rnn_hid_dim, num_layers=rnn_num_layers, bidirectional=False, dropout=rnn_dropout)
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 3)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))
        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        outputs = self.output(rnn_output)
        return outputs, rnn_state
