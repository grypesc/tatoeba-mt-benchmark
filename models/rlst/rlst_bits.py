import random
import torch

import torch.nn as nn


torch.set_printoptions(threshold=10_000)
random.seed(20)
torch.manual_seed(20)


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

        self.rnn = nn.GRU(src_embed_dim + trg_embed_dim + 4, rnn_hid_dim, num_layers=rnn_num_layers, dropout=0.0)
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.activation = nn.LeakyReLU()
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 3)

    def forward(self, src, previous_output, bits, rnn_state):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))
        rnn_input = torch.cat((src_embedded, trg_embedded, bits.unsqueeze(0)), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        leaky_out = self.activation(self.linear(rnn_output))
        leaky_out = self.rnn_dropout(leaky_out)
        outputs = self.output(leaky_out)
        return outputs, rnn_state


class RLSTBits(nn.Module):
    """
    This is a both version of RLST that has additional 4 bits entering approximator.
    0 - Start of sequence
    1 - Already read EOS
    2 - Written a token in previous t
    3 - Read a token in previous t
    """

    def __init__(self, net, device, testing_episode_max_time, trg_vocab_len, discount, m,
                 src_eos_index, src_null_index, src_pad_index, trg_eos_index, trg_null_index, trg_pad_index):
        super().__init__()
        self.net = net
        self.device = device
        self.testing_episode_max_time = testing_episode_max_time
        self.trg_vocab_len = trg_vocab_len
        self.DISCOUNT = discount
        self.M = m  # Read after eos punishment

        self.SRC_EOS = torch.tensor([src_eos_index], device=device)
        self.SRC_NULL = torch.tensor([src_null_index], device=device)
        self.SRC_PAD = torch.tensor([src_pad_index], device=device)
        self.TRG_EOS = torch.tensor([trg_eos_index], device=device)
        self.TRG_NULL = torch.tensor([trg_null_index], device=device)
        self.TRG_PAD = torch.tensor([trg_pad_index], device=device)

        self.mistranslation_loss_per_word = nn.CrossEntropyLoss(ignore_index=int(self.TRG_PAD), reduction='none')

    def forward(self, src, trg=None, epsilon=0, teacher_forcing=0):
        if self.training:
            return self._training_episode(src, trg, epsilon, teacher_forcing)
        return self._testing_episode(src)

    def _training_episode(self, src, trg, epsilon, teacher_forcing):
        device = self.device
        batch_size = src.size()[1]
        src_seq_len = src.size()[0]
        trg_seq_len = trg.size()[0]
        word_output = torch.full((1, batch_size), int(self.TRG_NULL), device=device)
        rnn_state = torch.zeros((self.net.rnn_num_layers, batch_size, self.net.rnn_hid_dim), device=device)

        word_outputs = torch.zeros((trg_seq_len, batch_size, self.trg_vocab_len), device=device)
        Q_used = torch.zeros((src_seq_len + trg_seq_len - 1, batch_size), device=device)
        Q_target = torch.zeros((src_seq_len + trg_seq_len - 1, batch_size), device=device)

        writing_agents = torch.full((1, batch_size), False, device=device, requires_grad=False)
        naughty_agents = torch.full((1, batch_size,), False, device=device, requires_grad=False)  # Want more input after input eos
        terminated_agents = torch.full((1, batch_size,), False, device=device, requires_grad=False)

        bits = torch.zeros((batch_size, 4), device=device)
        bits[:, 0] = 1.0
        bits[:, 3] = 1.0

        i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # input indices
        j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # output indices
        t = 0  # time
        actions_count = torch.zeros(3, dtype=torch.long, device=device, requires_grad=False)

        while True:
            input = torch.gather(src, 0, i)
            input[writing_agents] = self.SRC_NULL
            input[naughty_agents] = self.SRC_EOS
            output, rnn_state = self.net(input, word_output, bits, rnn_state)
            _, word_output = torch.max(output[:, :, :-3], dim=2)
            action = torch.max(output[:, :, -3:], 2)[1]

            random_action_agents = torch.rand((1, batch_size), device=device) < epsilon
            random_action = torch.randint(low=0, high=3, size=(1, batch_size), device=device)
            action[random_action_agents] = random_action[random_action_agents]

            Q_used[t, :] = torch.gather(output[0, :, -3:], 1, action.T).squeeze_(1)
            Q_used[t, terminated_agents.squeeze(0)] = 0

            with torch.no_grad():
                reading_agents = ~terminated_agents * (action == 0)
                writing_agents = ~terminated_agents * (action == 1)
                bothing_agents = ~terminated_agents * (action == 2)

                actions_count[0] += reading_agents.sum()
                actions_count[1] += writing_agents.sum()
                actions_count[2] += bothing_agents.sum()

            agents_outputting = writing_agents + bothing_agents
            word_outputs[j[agents_outputting], agents_outputting.squeeze(0), :] = output[0, agents_outputting.squeeze(0), :-3]

            just_terminated_agents = agents_outputting * (torch.gather(trg, 0, j) == self.TRG_EOS).squeeze_(0)
            naughty_agents = (reading_agents + bothing_agents) * (torch.gather(src, 0, i) == self.SRC_EOS).squeeze_(0)
            i = i + ~naughty_agents * (reading_agents + bothing_agents)
            old_j = j
            j = j + agents_outputting

            terminated_agents = terminated_agents + just_terminated_agents

            i[i >= src_seq_len] = src_seq_len - 1
            j[j >= trg_seq_len] = trg_seq_len - 1

            if random.random() < teacher_forcing:
                word_output = torch.gather(trg, 0, old_j)
            word_output[reading_agents] = self.TRG_NULL

            bits = torch.zeros((batch_size, 4), device=device)
            bits[torch.gather(src, 0, i).squeeze(0) == self.SRC_EOS, 1] = 1.0
            bits[agents_outputting.squeeze(0), 2] = 1.0
            bits[(reading_agents + bothing_agents).squeeze(0), 3] = 1.0

            with torch.no_grad():
                _input = torch.gather(src, 0, i)
                _input[writing_agents] = self.SRC_NULL
                _input[naughty_agents] = self.SRC_EOS
                _output, _ = self.net(_input, word_output, bits, rnn_state)
                next_best_action_value, _ = torch.max(_output[:, :, -3:], 2)

                reward = (-1) * self.mistranslation_loss_per_word(output[0, :, :-3], torch.gather(trg, 0, old_j)[0, :]).unsqueeze(0)
                Q_target[t, :] = reward + self.DISCOUNT * next_best_action_value
                Q_target[t, terminated_agents.squeeze(0)] = 0
                Q_target[t, reading_agents.squeeze(0)] = next_best_action_value[reading_agents]
                Q_target[t, (reading_agents * naughty_agents).squeeze(0)] = self.DISCOUNT * next_best_action_value[reading_agents * naughty_agents]
                Q_target[t, just_terminated_agents.squeeze(0)] = reward[just_terminated_agents]
                Q_target[t, naughty_agents.squeeze(0)] -= self.M

                if terminated_agents.all() or t >= src_seq_len + trg_seq_len - 2:
                    return word_outputs, Q_used, Q_target, actions_count.unsqueeze_(dim=1)
            t += 1

    def _testing_episode(self, src):
        device = self.device
        batch_size = src.size()[1]
        src_seq_len = src.size()[0]
        word_output = torch.full((1, batch_size), int(self.TRG_NULL), device=device)
        rnn_state = torch.zeros((self.net.rnn_num_layers, batch_size, self.net.rnn_hid_dim), device=device)

        word_outputs = torch.zeros((self.testing_episode_max_time, batch_size, self.trg_vocab_len), device=device)

        writing_agents = torch.full((1, batch_size), False, device=device, requires_grad=False)
        naughty_agents = torch.full((1, batch_size,), False, device=device, requires_grad=False)  # Want more input after input eos
        after_eos_agents = torch.full((1, batch_size,), False, device=device, requires_grad=False)  # Already outputted EOS

        i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # input indices
        j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # output indices
        t = 0  # time
        actions_count = torch.zeros(3, dtype=torch.long, device=device, requires_grad=False)

        bits = torch.zeros((batch_size, 4), device=device)
        bits[:, 0] = 1.0
        bits[:, 3] = 1.0

        while True:
            input = torch.gather(src, 0, i)
            input[writing_agents] = self.SRC_NULL
            input[naughty_agents] = self.SRC_EOS
            output, rnn_state = self.net(input, word_output, bits, rnn_state)
            _, word_output = torch.max(output[:, :, :-3], dim=2)
            action = torch.max(output[:, :, -3:], 2)[1]

            reading_agents = (action == 0)
            writing_agents = (action == 1)
            bothing_agents = (action == 2)

            actions_count[0] += (~after_eos_agents * reading_agents).sum()
            actions_count[1] += (~after_eos_agents * writing_agents).sum()
            actions_count[2] += (~after_eos_agents * bothing_agents).sum()

            agents_outputting = writing_agents + bothing_agents
            word_outputs[j[agents_outputting], agents_outputting.squeeze(0), :] = output[0, agents_outputting.squeeze(0), :-3]

            after_eos_agents += (word_output == self.TRG_EOS)
            naughty_agents = (reading_agents + bothing_agents) * (torch.gather(src, 0, i) == self.SRC_EOS).squeeze_(0)
            i = i + ~naughty_agents * (reading_agents + bothing_agents)
            j = j + agents_outputting

            i[i >= src_seq_len] = src_seq_len - 1
            word_output[reading_agents] = self.TRG_NULL

            bits = torch.zeros((batch_size, 4), device=device)
            bits[torch.gather(src, 0, i).squeeze(0) == self.SRC_EOS, 1] = 1.0
            bits[agents_outputting.squeeze(0), 2] = 1.0
            bits[(reading_agents + bothing_agents).squeeze(0), 3] = 1.0

            if t >= self.testing_episode_max_time - 1:
                return word_outputs, None, None, actions_count.unsqueeze_(dim=1)
            t += 1