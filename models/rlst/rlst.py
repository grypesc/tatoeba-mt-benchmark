import random
import torch

import torch.nn as nn

torch.set_printoptions(threshold=10_000)
random.seed(20)
torch.manual_seed(20)


class LeakyNet(nn.Module):
    """Simple GRU approximator for RLST. Suffers from gradient vanishing if has too many layers."""
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

        self.rnn = nn.GRU(src_embed_dim + trg_embed_dim, rnn_hid_dim, num_layers=rnn_num_layers, dropout=0.0, batch_first=True)
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.activation = nn.LeakyReLU()
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 2)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))
        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        leaky_out = self.activation(self.linear(rnn_output))
        leaky_out = self.rnn_dropout(leaky_out)
        outputs = self.output(leaky_out)
        return outputs, rnn_state


class LeakyResidualApproximator(nn.Module):
    """Residual approximator for RLST. 'Simply the best.' - Tina Turner
    Currently does not support pretrained embeddings."""
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
        self.rnns = nn.ModuleList([nn.GRU(rnn_hid_dim, rnn_hid_dim, batch_first=True) for _ in range(rnn_num_layers)])
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.activation = nn.LeakyReLU()
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 2)

    def forward(self, src, previous_output, rnn_states):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))

        rnn_input = self.activation(self.embedding_linear(torch.cat((src_embedded, trg_embedded), dim=2)))
        rnn_input = self.embedding_dropout(rnn_input)
        rnn_new_states = torch.zeros(rnn_states.size(), device=src_embedded.device)
        for i, rnn in enumerate(self.rnns):
            rnn_out, rnn_new_states[i, :] = rnn(rnn_input, rnn_states[i:i + 1])
            rnn_input = rnn_out + rnn_input

        leaky_output = self.rnn_dropout(self.activation(self.linear(rnn_input)))
        outputs = self.output(leaky_output)
        return outputs, rnn_new_states


class LeakyResidualNormApproximator(nn.Module):
    """Residual approximator for RLST. 'Simply the best.' - Tina Turner
    Currently does not support pretrained embeddings."""
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
        self.rnns = nn.ModuleList([nn.GRU(rnn_hid_dim, rnn_hid_dim, batch_first=True) for _ in range(rnn_num_layers)])
        self.norm = nn.ModuleList(nn.LayerNorm(rnn_hid_dim, rnn_hid_dim) for _ in range(rnn_num_layers + 1))
        self.linear = nn.Linear(rnn_hid_dim, rnn_hid_dim)
        self.activation = nn.LeakyReLU()
        self.output = nn.Linear(rnn_hid_dim, len(trg_vocab) + 2)

    def forward(self, src, previous_output, rnn_states):
        src_embedded = self.embedding_dropout(self.src_embedding(src))
        trg_embedded = self.embedding_dropout(self.trg_embedding(previous_output))

        rnn_input = self.norm[0](self.embedding_linear(torch.cat((src_embedded, trg_embedded), dim=2)))
        rnn_input = self.embedding_dropout(self.activation(rnn_input))
        rnn_new_states = torch.zeros(rnn_states.size(), device=src_embedded.device)
        for i, rnn in enumerate(self.rnns):
            rnn_out, rnn_new_states[i, :] = rnn(rnn_input, rnn_states[i:i + 1])
            rnn_input = rnn_out + rnn_input
            rnn_input = self.norm[i + 1](rnn_input)

        leaky_output = self.rnn_dropout(self.activation(self.linear(rnn_input)))
        outputs = self.output(leaky_output)
        return outputs, rnn_new_states


class RLST(nn.Module):
    """
    This class implements RLST algorithm presented in the paper. Given batch size of n, it creates n partially observable
    training or testing environments in which n interpreter agents operate in order to transform source sequences into the target ones.
    At time t each agent can be at different indices in input and output sequences, this indices are vectors i and j.
    At t=0 each agent is fed with first source token. If one decides to read, next token will be read in next t (t=1).
    Agent goes naughty if he decides to read after he has already read source EOS token.
    """

    def __init__(self, approximator, testing_episode_max_time, trg_vocab_len, discount, m,
                 src_eos_index, src_null_index, trg_eos_index, trg_null_index):
        super().__init__()
        self.approximator = approximator
        self.testing_episode_max_time = testing_episode_max_time
        self.trg_vocab_len = trg_vocab_len
        self.DISCOUNT = discount
        self.M = m  # Read after eos punishment

        self.SRC_EOS = src_eos_index
        self.SRC_NULL = src_null_index
        self.TRG_EOS = trg_eos_index
        self.TRG_NULL = trg_null_index

        self.mistranslation_loss_per_token = nn.CrossEntropyLoss(reduction='none')

    def forward(self, src, trg=None, epsilon=0, teacher_forcing=0):
        if self.training:
            return self._training_episode(src, trg, epsilon, teacher_forcing)
        return self._testing_episode(src)

    def _training_episode(self, src, trg, epsilon, teacher_forcing):
        """
        :param src: Tensor of shape batch size x src seq length
        :param trg: Tensor of shape batch size x trg seq length
        :param epsilon: Probability of random action in epsilon greedy strategy
        :param teacher_forcing: Probability of output being ground truth at each time step
        :return: token_probs: Tensor of shape batch size x trg seq len x number of features e.g. target vocab length
        :return: Q_used: Tensor of shape batch size x time . Containes Q values of actions taken by agents
        :return: Q_target: Tensor of shape batch size x time. Containes best Q values in next time step w.r.t Q_used
        :return: actions_count: Tensor of shape 3. Contains number of actions taken by agents: read, write
        """

        device = src.device
        batch_size = src.size()[0]
        src_seq_len = src.size()[1]
        trg_seq_len = trg.size()[1]
        word_output = torch.full((batch_size, 1), int(self.TRG_NULL), device=device)
        rnn_state = torch.zeros((self.approximator.rnn_num_layers, batch_size, self.approximator.rnn_hid_dim), device=device)

        token_probs = torch.zeros((batch_size, trg_seq_len, self.trg_vocab_len), device=device)
        Q_used = torch.zeros((batch_size, src_seq_len + trg_seq_len - 1), device=device)
        Q_target = torch.zeros((batch_size, src_seq_len + trg_seq_len - 1), device=device)

        terminated_agents = torch.full((batch_size, 1), False, device=device)

        i = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=device)  # input indices
        j = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=device)  # output indices
        actions_count = torch.zeros(2, dtype=torch.long, device=device)

        input = torch.gather(src, 1, i)
        output, rnn_state = self.approximator(input, word_output, rnn_state)
        action = torch.max(output[:, :, -2:], 2)[1]

        for t in range(src_seq_len + trg_seq_len - 1):
            _, word_output = torch.max(output[:, :, :-2], dim=2)
            random_action_agents = torch.rand((batch_size, 1), device=device) < epsilon
            random_action = torch.randint(low=0, high=2, size=(batch_size, 1), device=device)
            action[random_action_agents] = random_action[random_action_agents]

            Q_used[:, t] = torch.gather(output[:, 0, -2:], 1, action).squeeze_(1)
            Q_used[terminated_agents.squeeze(1), t] = 0

            with torch.no_grad():
                reading_agents = ~terminated_agents * (action == 0)
                writing_agents = ~terminated_agents * (action == 1)

                actions_count[0] += reading_agents.sum()
                actions_count[1] += writing_agents.sum()

                just_terminated_agents = writing_agents * (torch.gather(trg, 1, j) == self.TRG_EOS)
                naughty_agents = reading_agents * (torch.gather(src, 1, i) == self.SRC_EOS)
                i = i + ~naughty_agents * reading_agents
                old_j = j
                j = j + writing_agents * ~just_terminated_agents
                terminated_agents = terminated_agents + just_terminated_agents

                if random.random() < teacher_forcing:
                    word_output = torch.gather(trg, 1, old_j)
                word_output[reading_agents] = self.TRG_NULL

                reward = (-1) * self.mistranslation_loss_per_token(output[:, 0, :-2], torch.gather(trg, 1, old_j)[:, 0])

            token_probs[writing_agents.squeeze(1), old_j[writing_agents], :] = output[writing_agents.squeeze(1), 0, :-2]

            input = torch.gather(src, 1, i)
            input[writing_agents] = self.SRC_NULL
            input[naughty_agents] = self.SRC_EOS
            output, rnn_state = self.approximator(input, word_output, rnn_state)
            next_best_action_value, action = torch.max(output[:, :, -2:], 2)
            next_best_action_value = next_best_action_value.squeeze_(1)

            with torch.no_grad():
                Q_target[:, t] = reward + self.DISCOUNT * next_best_action_value
                Q_target[terminated_agents.squeeze(1), t] = 0
                Q_target[reading_agents.squeeze(1), t] = next_best_action_value[reading_agents.squeeze(1)]
                Q_target[naughty_agents.squeeze(1), t] = self.DISCOUNT * next_best_action_value[reading_agents.squeeze(1) * naughty_agents.squeeze(1)]
                Q_target[just_terminated_agents.squeeze(1), t] = reward[just_terminated_agents.squeeze(1)]
                Q_target[naughty_agents.squeeze(1), t] -= self.M

                if torch.all(terminated_agents):
                    return token_probs, Q_used, Q_target.detach_(), actions_count.unsqueeze_(dim=1)

        return token_probs, Q_used, Q_target.detach_(), actions_count.unsqueeze_(dim=1)

    def _testing_episode(self, src):
        """
        :param src: Tensor of shape batch size x testing_episode_max_time
        :return: token_probs: Tensor of shape batch size x trg seq len x number of features e.g. target vocab length
        :return: None
        :return: None
        :return: actions_count: Tensor of shape 2. Contains number of actions taken by agents: read, write
        """

        device = src.device
        batch_size = src.size()[0]
        src_seq_len = src.size()[1]
        word_output = torch.full((batch_size, 1), int(self.TRG_NULL), device=device)
        rnn_state = torch.zeros((self.approximator.rnn_num_layers, batch_size, self.approximator.rnn_hid_dim), device=device)

        token_probs = torch.zeros((batch_size, self.testing_episode_max_time, self.trg_vocab_len), device=device)

        writing_agents = torch.full((batch_size, 1), False, device=device)
        naughty_agents = torch.full((batch_size, 1), False, device=device)  # Want more input after input eos
        after_eos_agents = torch.full((batch_size, 1), False, device=device)  # Already outputted EOS

        i = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=device)  # input indices
        j = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=device)  # output indices
        actions_count = torch.zeros(2, dtype=torch.long, device=device)

        for t in range(self.testing_episode_max_time):
            input = torch.gather(src, 1, i)
            input[writing_agents] = self.SRC_NULL
            input[naughty_agents] = self.SRC_EOS
            output, rnn_state = self.approximator(input, word_output, rnn_state)
            _, word_output = torch.max(output[:, :, :-2], dim=2)
            action = torch.max(output[:, :, -2:], 2)[1]

            reading_agents = (action == 0)
            writing_agents = (action == 1)

            actions_count[0] += (~after_eos_agents * reading_agents).sum()
            actions_count[1] += (~after_eos_agents * writing_agents).sum()

            token_probs[writing_agents.squeeze(1), j[writing_agents], :] = output[writing_agents.squeeze(1), 0, :-2]

            after_eos_agents += (word_output == self.TRG_EOS)
            naughty_agents = reading_agents * (torch.gather(src, 1, i) == self.SRC_EOS)
            i = i + ~naughty_agents * reading_agents
            j = j + writing_agents

            i[i >= src_seq_len] = src_seq_len - 1
            word_output[reading_agents] = self.TRG_NULL

        return token_probs, None, None, actions_count.unsqueeze_(dim=1)



