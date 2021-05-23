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


class LeakyNet(nn.Module):
    def __init__(self,
                 src_vocab,
                 trg_vocab,
                 use_pretrained_embeddings,
                 rnn_hid_dim,
                 rnn_dropout,
                 rnn_num_layers,
                 src_embed_dim=256,
                 trg_embed_dim=256,
                 embedding_dropout=0.0,
                 ):
        super().__init__()

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_num_layers = rnn_num_layers

        self.src_embedding = nn.Embedding(len(src_vocab), src_embed_dim)
        self.trg_embedding = nn.Embedding(len(trg_vocab), trg_embed_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        if use_pretrained_embeddings:
            src_embed_dim = src_vocab.vectors.size()[1]
            trg_embed_dim = trg_vocab.vectors.size()[1]
            self.src_embedding = nn.Embedding(len(src_vocab), src_embed_dim).from_pretrained(src_vocab.vectors, freeze=True)
            self.trg_embedding = nn.Embedding(len(trg_vocab), trg_embed_dim).from_pretrained(trg_vocab.vectors, freeze=True)

        self.rnn = nn.GRU(src_embed_dim + trg_embed_dim, rnn_hid_dim, num_layers=rnn_num_layers, dropout=0.0)
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.activation = nn.LeakyReLU()
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 3)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))
        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        leaky_out = self.activation(self.linear(rnn_output))
        leaky_out = self.rnn_dropout(leaky_out)
        outputs = self.output(leaky_out)
        return outputs, rnn_state


class MoneyShot(nn.Module):

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
        self.src_leaky = nn.Linear(src_embed_dim, rnn_hid_dim)
        self.trg_embedding = nn.Embedding(len(trg_vocab), trg_embed_dim)
        self.trg_leaky = nn.Linear(trg_embed_dim, rnn_hid_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.rnns = nn.ModuleList([nn.GRU(rnn_hid_dim, rnn_hid_dim) for _ in range(rnn_num_layers)])
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.activation = nn.LeakyReLU()
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 3)

    def forward(self, src, previous_output, rnn_states):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        src_embedded = self.activation(self.src_leaky(src_embedded))
        src_embedded = self.rnn_dropout(src_embedded)

        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))
        trg_embedded = self.activation(self.trg_leaky(trg_embedded))
        trg_embedded = self.rnn_dropout(trg_embedded)

        rnn_input = src_embedded
        rnn_new_states = torch.zeros(rnn_states.size(), device=src_embedded.device)
        res_out = None
        for i, rnn in enumerate(self.rnns):
            res_out, rnn_new_states[i, :] = self._skip_rep(rnn_input, rnn, rnn_states[i:i + 1])
            rnn_input = res_out
            if self.rnn_num_layers / (i + 1) == 2:
                rnn_input += trg_embedded

        leaky_output = self.rnn_dropout(self.activation(self.linear(res_out)))
        outputs = self.output(leaky_output)
        return outputs, rnn_new_states

    @staticmethod
    def _skip_rep(input, rnn, rnn_state):
        rnn_output, rnn_new_state = rnn(input, rnn_state)
        return input + rnn_output, rnn_new_state


class ResidualApproximator(nn.Module):

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
        assert src_embed_dim + trg_embed_dim == rnn_hid_dim
        self.rnns = nn.ModuleList([nn.GRU(rnn_hid_dim, rnn_hid_dim) for _ in range(rnn_num_layers)])
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 3)

    def forward(self, src, previous_output, rnn_states):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))

        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_new_states = torch.zeros(rnn_states.size(), device=src_embedded.device)
        res_out = None
        for i, rnn in enumerate(self.rnns):
            res_out, rnn_new_states[i, :] = self._skip_rep(rnn_input, rnn, rnn_states[i:i + 1])
            res_out = self.rnn_dropout(res_out)
            rnn_input = res_out

        outputs = self.output(res_out)
        return outputs, rnn_new_states

    @staticmethod
    def _skip_rep(input, rnn, rnn_state):
        rnn_output, rnn_new_state = rnn(input, rnn_state)
        return input + rnn_output, rnn_new_state


class LeakyResidualApproximator(nn.Module):

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
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.embedding_linear = nn.Linear(src_embed_dim + trg_embed_dim, rnn_hid_dim)
        self.rnns = nn.ModuleList([nn.GRU(rnn_hid_dim, rnn_hid_dim) for _ in range(rnn_num_layers)])
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.activation = nn.LeakyReLU()
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 3)

    def forward(self, src, previous_output, rnn_states):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))

        rnn_input = self.activation(self.embedding_linear(torch.cat((src_embedded, trg_embedded), dim=2)))
        rnn_new_states = torch.zeros(rnn_states.size(), device=src_embedded.device)
        res_out = None
        for i, rnn in enumerate(self.rnns):
            res_out, rnn_new_states[i, :] = self._skip_rep(rnn_input, rnn, rnn_states[i:i + 1])
            rnn_input = res_out

        leaky_output = self.rnn_dropout(self.activation(self.linear(res_out)))
        outputs = self.output(leaky_output)
        return outputs, rnn_new_states

    @staticmethod
    def _skip_rep(input, rnn, rnn_state):
        rnn_output, rnn_new_state = rnn(input, rnn_state)
        return input + rnn_output, rnn_new_state


class ResidualApproximator9000(nn.Module):
    """Bad boy with policy coming straight from GRU hidden states."""

    def __init__(self,
                 src_vocab,
                 trg_vocab,
                 use_pretrained_embeddings,
                 rnn_hid_dim,
                 rnn_dropout,
                 rnn_num_layers,
                 src_embed_dim=256,
                 trg_embed_dim=256,
                 embedding_dropout=0.0,
                 policy_dropout=0.0):
        super().__init__()

        self.rnn_hid_dim = rnn_hid_dim
        self.rnn_num_layers = rnn_num_layers
        self.src_embedding = nn.Embedding(len(src_vocab), src_embed_dim)
        self.trg_embedding = nn.Embedding(len(trg_vocab), trg_embed_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        assert src_embed_dim + trg_embed_dim == rnn_hid_dim
        self.rnns = nn.ModuleList([nn.GRU(rnn_hid_dim, rnn_hid_dim) for _ in range(rnn_num_layers)])
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.policy_dropout = nn.Dropout(policy_dropout)
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab))
        self.policy_output = nn.Linear(rnn_num_layers * rnn_hid_dim, 3)

    def forward(self, src, previous_output, rnn_states):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))

        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_new_states = torch.zeros(rnn_states.size(), device=src_embedded.device)
        res_out = None

        for i, rnn in enumerate(self.rnns):
            res_out, rnn_new_state = self._skip_rep(rnn_input, rnn, rnn_states[i:i + 1])
            res_out = self.rnn_dropout(res_out)
            rnn_input = res_out
            rnn_new_states[i, :] = rnn_new_state

        word_outputs = self.output(res_out)
        policy_output = self.policy_output(rnn_new_states.reshape(rnn_new_states.size()[1], self.rnn_num_layers * self.rnn_hid_dim)).unsqueeze_(0)
        policy_output = self.policy_dropout(policy_output)
        outputs = torch.cat((word_outputs, policy_output), dim=2)
        return outputs, rnn_new_states

    def _skip_rep(self, input, rnn, rnn_state):
        rnn_output, rnn_new_state = rnn(input, rnn_state)
        return input + rnn_output, rnn_new_state
