import argparse
import math
import os
import random
import time
import torch

import torch.nn as nn
import torch.optim as optim

from utils.data_pipeline import DataPipeline
from utils.tools import epoch_time, actions_ratio, save_model, BleuScorer, parse_utils
from criterions.rlst_criterion import RLSTCriterion

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

        self.rnn = nn.GRU(src_embed_dim + trg_embed_dim, rnn_hid_dim, num_layers=rnn_num_layers, dropout=0.0)
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


class RLST(nn.Module):
    """
    This class implements RLST algorithm presented in the paper. Given batch size of n, it creates n partially observable
    training or testing environments in which n agents operate in order to transform source sequences into the target ones.
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

        i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # input indices
        j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device, requires_grad=False)  # output indices
        t = 0  # time
        actions_count = torch.zeros(3, dtype=torch.long, device=device, requires_grad=False)

        while True:
            input = torch.gather(src, 0, i)
            input[writing_agents] = self.SRC_NULL
            input[naughty_agents] = self.SRC_EOS
            output, rnn_state = self.net(input, word_output, rnn_state)
            _, word_output = torch.max(output[:, :, :-2], dim=2)
            action = torch.max(output[:, :, -2:], 2)[1]

            random_action_agents = torch.rand((1, batch_size), device=device) < epsilon
            random_action = torch.randint(low=0, high=2, size=(1, batch_size), device=device)
            action[random_action_agents] = random_action[random_action_agents]

            Q_used[t, :] = torch.gather(output[0, :, -2:], 1, action.T).squeeze_(1)
            Q_used[t, terminated_agents.squeeze(0)] = 0

            with torch.no_grad():
                reading_agents = ~terminated_agents * (action == 0)
                writing_agents = ~terminated_agents * (action == 1)
                bothing_agents = ~terminated_agents * (action == 2)

                actions_count[0] += reading_agents.sum()
                actions_count[1] += writing_agents.sum()
                actions_count[2] += bothing_agents.sum()

            agents_outputting = writing_agents + bothing_agents
            word_outputs[j[agents_outputting], agents_outputting.squeeze(0), :] = output[0, agents_outputting.squeeze(0), :-2]

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

            with torch.no_grad():
                _input = torch.gather(src, 0, i)
                _input[writing_agents] = self.SRC_NULL
                _input[naughty_agents] = self.SRC_EOS
                _output, _ = self.net(_input, word_output, rnn_state)
                next_best_action_value, _ = torch.max(_output[:, :, -2:], 2)

                reward = (-1) * self.mistranslation_loss_per_word(output[0, :, :-2], torch.gather(trg, 0, old_j)[0, :]).unsqueeze(0)
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

        while True:
            input = torch.gather(src, 0, i)
            input[writing_agents] = self.SRC_NULL
            input[naughty_agents] = self.SRC_EOS
            output, rnn_state = self.net(input, word_output, rnn_state)
            _, word_output = torch.max(output[:, :, :-2], dim=2)
            action = torch.max(output[:, :, -2:], 2)[1]

            reading_agents = (action == 0)
            writing_agents = (action == 1)
            bothing_agents = (action == 2)

            actions_count[0] += (~after_eos_agents * reading_agents).sum()
            actions_count[1] += (~after_eos_agents * writing_agents).sum()
            actions_count[2] += (~after_eos_agents * bothing_agents).sum()

            agents_outputting = writing_agents + bothing_agents
            word_outputs[j[agents_outputting], agents_outputting.squeeze(0), :] = output[0, agents_outputting.squeeze(0), :-2]

            after_eos_agents += (word_output == self.TRG_EOS)
            naughty_agents = (reading_agents + bothing_agents) * (torch.gather(src, 0, i) == self.SRC_EOS).squeeze_(0)
            i = i + ~naughty_agents * (reading_agents + bothing_agents)
            j = j + agents_outputting

            i[i >= src_seq_len] = src_seq_len - 1
            word_output[reading_agents] = self.TRG_NULL

            if t >= self.testing_episode_max_time - 1:
                return word_outputs, None, None, actions_count.unsqueeze_(dim=1)
            t += 1


def train_epoch(optimizer, epsilon, teacher_forcing, clip):
    model.train()
    rlst_criterion.train()
    epoch_mistranslation_loss = 0
    epoch_policy_loss = 0
    policy_multiplier = None
    total_actions = torch.zeros((3, 1), dtype=torch.long, device=device)
    for iteration, (src, trg) in enumerate(train_loader, 1):
        src, trg = src.to(device), trg.to(device)
        word_outputs, Q_used, Q_target, actions = model(src, trg, epsilon, teacher_forcing)
        total_actions += actions.cumsum(dim=1)
        optimizer.zero_grad()
        word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
        trg = trg.view(-1)

        loss, mistranslation_loss, policy_loss, policy_multiplier = rlst_criterion(word_outputs, trg, Q_used, Q_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_mistranslation_loss += mistranslation_loss.item()
        epoch_policy_loss += policy_loss.item()
    return epoch_mistranslation_loss / len(train_loader), epoch_policy_loss / len(train_loader), total_actions.squeeze(1).tolist(), policy_multiplier


def evaluate_epoch(loader, bleu_scorer):
    model.eval()
    rlst_criterion.eval()
    epoch_loss, epoch_bleu = 0, 0
    total_actions = torch.zeros((3, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        for iteration, (src, trg) in enumerate(loader):
            src, trg = src.to(device), trg.to(device)
            word_outputs, _, _, actions = model(src)
            total_actions += actions.cumsum(dim=1)
            bleu_scorer.register_minibatch(word_outputs, trg)
            word_outputs_clipped = word_outputs[:trg.size()[0], :, :]
            word_outputs_clipped = word_outputs_clipped.view(-1, word_outputs_clipped.shape[-1])
            trg = trg.view(-1)
            _, _mistranslation_loss, _, _ = rlst_criterion(word_outputs_clipped, trg, 0, 0)
            epoch_loss += _mistranslation_loss.item()
    return epoch_loss / len(loader), bleu_scorer.epoch_score(), total_actions.squeeze(1).tolist()


def parse_args():
    parser = argparse.ArgumentParser()
    parse_utils(parser)
    parser.add_argument('--testing-episode-max-time',
                        help='maximum episode time during testing after which agents are terminated, '
                             'if too low it will disallow agents to transform long sequences',
                        type=int,
                        default=64)
    parser.add_argument('--rnn-hid-dim',
                        help='approximator\'s rnn hidden size',
                        type=int,
                        default=512)
    parser.add_argument('--rnn-num-layers',
                        help='number of rnn layers',
                        type=int,
                        default=2)
    parser.add_argument('--rnn-dropout',
                        help='dropout between rnn layers',
                        type=float,
                        default=0.00)
    parser.add_argument('--discount',
                        help='discount',
                        type=float,
                        default=0.90)
    parser.add_argument('--epsilon',
                        help='epsilon for epsilon-greedy strategy',
                        type=float,
                        default=0.15)
    parser.add_argument('--teacher-forcing',
                        help='teacher forcing',
                        type=float,
                        default=0.5)
    parser.add_argument('--M',
                        help='punishment for reading after reading eos',
                        type=float,
                        default=3.0)
    parser.add_argument('--N',
                        help='estimated number of training mini batches after which policy loss multiplier will be close '
                        'to its asymptote/maximum value',
                        type=float,
                        default=50_000)
    parser.add_argument('--eta-min',
                        help='minimum eta value',
                        type=float,
                        default=0.02)
    parser.add_argument('--eta-max',
                        help='eta maximum value, its asymptote',
                        type=float,
                        default=0.2)
    parser.add_argument('--rho',
                        help='rho for moving exponential average of losses weights',
                        type=float,
                        default=0.99)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    data = DataPipeline(batch_size=args.batch_size, src_lang=args.src, trg_lang=args.trg, null_replaces_bos=True,
                        token_min_freq=args.token_min_freq, use_pretrained_embeds=args.use_pretrained_embeddings)
    src_vocab = data.src_vocab
    trg_vocab = data.trg_vocab
    train_loader = data.train_loader
    valid_loader = data.valid_loader
    test_loader = data.test_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = LeakyNet(src_vocab, trg_vocab, args.use_pretrained_embeddings, args.rnn_hid_dim, args.rnn_dropout, args.rnn_num_layers,
                               args.src_embed_dim, args.trg_embed_dim, args.embed_dropout).to(device)
    if args.load_model_name:
        net.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.load_model_name)))
    model = RLST(net, device, args.testing_episode_max_time, len(trg_vocab), args.discount, args.M,
                 src_vocab.stoi['<eos>'],
                 src_vocab.stoi['<null>'],
                 src_vocab.stoi['<pad>'],
                 trg_vocab.stoi['<eos>'],
                 trg_vocab.stoi['<null>'],
                 trg_vocab.stoi['<pad>'])

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rlst_criterion = RLSTCriterion(args.rho, trg_vocab.stoi['<pad>'], args.N, args.eta_min, args.eta_max)

    print(vars(args))
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters\n')
    print(net)
    
    bleu_scorer = BleuScorer(trg_vocab, device)
    best_val_bleu = 0.0

    if not args.test:
        for epoch in range(args.epochs):
            start_time = time.time()
            train_loss, policy_loss, train_actions, last_policy_multiplier = train_epoch(optimizer, args.epsilon, args.teacher_forcing, args.clip)
            val_loss, val_bleu, val_actions = evaluate_epoch(valid_loader, bleu_scorer)

            save_model(net, args.checkpoint_dir, "rlst", val_bleu > best_val_bleu)
            best_val_bleu = val_bleu if val_bleu > best_val_bleu else best_val_bleu

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print('Train loss: {}, PPL: {}, policy loss: {}, eta: {}, epsilon: {}, action ratio: {}'
                  .format(round(train_loss, 3), round(math.exp(train_loss), 3), round(policy_loss, 3), round(last_policy_multiplier, 2), round(args.epsilon, 2), actions_ratio(train_actions)))
            print('Valid loss: {}, PPL: {}, BLEU: {}, action ratio: {}\n'.format(round(val_loss, 3), round(math.exp(val_loss), 3), round(100*val_bleu, 2), actions_ratio(val_actions)))

    else:
        test_loss, test_bleu, test_actions = evaluate_epoch(test_loader, bleu_scorer)
        print('Test loss: {}, PPL: {}, BLEU: {}, action ratio: {}'.format(round(test_loss, 5), round(math.exp(test_loss), 3), round(100*test_bleu, 2), actions_ratio(test_actions)))
        test_loss, test_bleu, test_actions = evaluate_epoch(data.long_test_loader, bleu_scorer)
        print('Test-long loss: {}, PPL: {}, BLEU: {}, action ratio: {}\n'.format(round(test_loss, 5), round(math.exp(test_loss), 3), round(100*test_bleu, 2), actions_ratio(test_actions)))

