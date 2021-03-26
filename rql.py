import math
import random
import time
import torch

import torch.nn as nn
import torch.optim as optim

from utils.data_pipeline_rql import DataPipelineRQL
from utils.tools import epoch_time, bleu, actions_ratio
from utils.rql_nets import Net, Net1, Net2

torch.set_printoptions(threshold=10_000)
random.seed(20)
torch.manual_seed(20)


class RQL(nn.Module):
    """
    This class implements RQL algorithm presented in the paper. Given batch size of n, it creates n partially observable
    training or testing environments in which n agents operate in order to transform source sequences into the target ones.
    """

    def __init__(self, net, device, testing_episode_time):
        super().__init__()
        self.net = net
        self.device = device
        self.testing_episode_time = testing_episode_time

    def forward(self, src, trg, epsilon, teacher_forcing):
        if self.training:
            return self._training_episode(src, trg, epsilon, teacher_forcing)
        return self._testing_episode(src)

    def _training_episode(self, src, trg, epsilon, teacher_forcing):
        device = self.device
        batch_size = src.size()[1]
        src_seq_len = src.size()[0]
        trg_seq_len = trg.size()[0]
        word_output = torch.full((1, batch_size), spa_vocab.stoi["<null>"], device=device)
        rnn_state = torch.zeros((NUM_RNN_LAYERS, batch_size, RNN_HID_DIM), device=device)

        word_outputs = torch.zeros((trg_seq_len, batch_size, len(spa_vocab)), device=device)
        Q_used = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)
        Q_target = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)

        writing_agents = torch.full((1, batch_size), False, device=device)
        naughty_agents = torch.full((1, batch_size,), False, device=device)  # Want more input after input eos
        terminated_agents = torch.full((1, batch_size,), False, device=device)

        i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)  # input indices
        j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)  # output indices
        t = 0  # time
        actions_count = torch.zeros(3, dtype=torch.long, device=device)

        while True:
            input = torch.gather(src, 0, i)
            input[writing_agents] = EN_NULL
            input[naughty_agents] = EN_PAD
            output, rnn_state = self.net(input, word_output, rnn_state)
            _, word_output = torch.max(output[:, :, :-3], dim=2)

            if random.random() < epsilon:
                action = torch.randint(low=0, high=3, size=(1, batch_size), device=device)
            else:
                action = torch.max(output[:, :, -3:], 2)[1]

            Q_used[t, :] = torch.gather(output[0, :, -3:], 1, action.T).squeeze_()
            Q_used[t, terminated_agents.squeeze()] = 0

            with torch.no_grad():
                reading_agents = ~terminated_agents * (action == 0)
                writing_agents = ~terminated_agents * (action == 1)
                bothing_agents = ~terminated_agents * (action == 2)

                actions_count[0] += reading_agents.sum()
                actions_count[1] += writing_agents.sum()
                actions_count[2] += bothing_agents.sum()

            agents_outputting = writing_agents + bothing_agents
            word_outputs[j[agents_outputting], agents_outputting.squeeze(), :] = output[0, agents_outputting.squeeze(), :-3]

            just_terminated_agents = agents_outputting * (torch.gather(trg, 0, j) == SPA_EOS).squeeze_()
            naughty_agents = (reading_agents + bothing_agents) * (torch.gather(src, 0, i) == EN_EOS).squeeze_()
            i = i + ~naughty_agents * (reading_agents + bothing_agents)
            old_j = j
            j = j + agents_outputting

            terminated_agents = terminated_agents + just_terminated_agents

            i[i >= src_seq_len] = src_seq_len - 1
            j[j >= trg_seq_len] = trg_seq_len - 1

            if random.random() < teacher_forcing:
                word_output = torch.gather(trg, 0, old_j)
            word_output[reading_agents] = SPA_NULL

            with torch.no_grad():
                _input = torch.gather(src, 0, i)
                _input[writing_agents] = EN_NULL
                _input[naughty_agents] = EN_PAD
                _output, _ = self.net(_input, word_output, rnn_state)
                next_best_action_value, _ = torch.max(_output[:, :, -3:], 2)

                reward = (-1) * mistranslation_loss_per_agent(output[0, :, :-3], torch.gather(trg, 0, old_j)[0, :]).unsqueeze(0)
                Q_target[t, :] = reward + DISCOUNT * next_best_action_value
                Q_target[t, terminated_agents.squeeze()] = 0
                Q_target[t, reading_agents.squeeze()] = next_best_action_value[reading_agents]
                Q_target[t, (reading_agents * naughty_agents).squeeze()] = DISCOUNT * next_best_action_value[reading_agents * naughty_agents]
                Q_target[t, just_terminated_agents.squeeze()] = reward[just_terminated_agents]
                Q_target[t, naughty_agents.squeeze()] -= 5.0

                if terminated_agents.all() or t >= src_seq_len + trg_seq_len - 1:
                    return word_outputs, Q_used, Q_target, actions_count
            t += 1

    def _testing_episode(self, src):
        device = self.device
        batch_size = src.size()[1]
        src_seq_len = src.size()[0]
        word_output = torch.full((1, batch_size), spa_vocab.stoi["<null>"], device=device)
        rnn_state = torch.zeros((NUM_RNN_LAYERS, batch_size, RNN_HID_DIM), device=device)

        word_outputs = torch.zeros((self.testing_episode_time, batch_size, len(spa_vocab)), device=device)

        writing_agents = torch.full((1, batch_size), False, device=device)
        naughty_agents = torch.full((1, batch_size,), False, device=device)  # Want more input after input eos
        after_eos_agents = torch.full((1, batch_size,), False, device=device)  # Already outputted EOS

        i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)  # input indices
        j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)  # output indices
        t = 0  # time
        actions_count = torch.zeros(3, dtype=torch.long, device=device)

        while True:
            input = torch.gather(src, 0, i)
            input[writing_agents] = EN_NULL
            input[naughty_agents] = EN_PAD
            output, rnn_state = self.net(input, word_output, rnn_state)
            _, word_output = torch.max(output[:, :, :-3], dim=2)
            action = torch.max(output[:, :, -3:], 2)[1]

            reading_agents = (action == 0)
            writing_agents = (action == 1)
            bothing_agents = (action == 2)

            actions_count[0] += (~after_eos_agents * reading_agents).sum()
            actions_count[1] += (~after_eos_agents * writing_agents).sum()
            actions_count[2] += (~after_eos_agents * bothing_agents).sum()

            agents_outputting = writing_agents + bothing_agents
            word_outputs[j[agents_outputting], agents_outputting.squeeze(), :] = output[0, agents_outputting.squeeze(), :-3]

            after_eos_agents += (word_output == SPA_EOS)
            naughty_agents = (reading_agents + bothing_agents) * (torch.gather(src, 0, i) == EN_EOS).squeeze_()
            i = i + ~naughty_agents * (reading_agents + bothing_agents)
            j = j + agents_outputting

            i[i >= src_seq_len] = src_seq_len - 1
            word_output[reading_agents] = SPA_NULL

            if t >= self.testing_episode_time - 1:
                return word_outputs, None, None, actions_count
            t += 1


def train_epoch(epsilon, teacher_forcing, ro_to_k, _mistranslation_loss_weight, _policy_loss_weight ):
    model.train()
    epoch_loss = 0

    total_actions = torch.zeros(3, dtype=torch.long, device=device)
    for iteration, (src, trg) in enumerate(train_loader, 1):
        ro_to_k *= RO
        w_k = (RO - ro_to_k) / (1 - ro_to_k)
        src, trg = src.to(device), trg.to(device)
        word_outputs, Q_used, Q_target, actions = model(src, trg, epsilon, teacher_forcing)
        no_eos_outputs = (word_outputs.max(2)[1] != trg) * (trg == SPA_EOS)
        total_actions += actions
        optimizer.zero_grad()
        word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
        trg = trg.view(-1)

        _policy_loss = policy_loss(Q_used, Q_target)
        _mistranslation_loss = mistranslation_loss_per_agent(word_outputs, trg)
        _mistranslation_loss[no_eos_outputs.view(-1)] *= NO_EOS_LOSS_MULTIPLIER
        _mistranslation_loss = torch.mean(_mistranslation_loss[trg != spa_vocab.stoi['<pad>']])
        _mistranslation_loss_weight = w_k * _mistranslation_loss_weight + (1 - w_k) * float(_mistranslation_loss)
        _policy_loss_weight = w_k * _policy_loss_weight + (1 - w_k) * float(_policy_loss)

        loss = _policy_loss / _policy_loss_weight + MISTRANSLATION_LOSS_MULTIPLIER * _mistranslation_loss / _mistranslation_loss_weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        epoch_loss += _mistranslation_loss.item()
    return epoch_loss / len(train_loader), total_actions.tolist(), ro_to_k, _mistranslation_loss_weight, _policy_loss_weight


def evaluate_epoch(loader):
    model.eval()
    epoch_loss, epoch_bleu = 0, 0
    total_actions = torch.zeros(3, dtype=torch.long, device=device)
    with torch.no_grad():
        for iteration, (src, trg) in enumerate(loader):
            src, trg = src.to(device), trg.to(device)
            word_outputs, _, _, actions = model(src, trg, 0, 0)
            total_actions += actions
            epoch_bleu += bleu(word_outputs, trg, spa_vocab, SPA_EOS, device)
            word_outputs_clipped = word_outputs[:trg.size()[0], :, :]
            word_outputs_clipped = word_outputs_clipped.view(-1, word_outputs_clipped.shape[-1])
            trg = trg.view(-1)
            epoch_loss += mistranslation_loss(word_outputs_clipped, trg).item()
    return epoch_loss / len(loader), epoch_bleu / len(loader), total_actions.tolist()


BATCH_SIZE = 64
N_EPOCHS = 30
RNN_HID_DIM = 256
DROPOUT = 0.0
NUM_RNN_LAYERS = 1
DISCOUNT = 0.99
MISTRANSLATION_LOSS_MULTIPLIER = 10
NO_EOS_LOSS_MULTIPLIER = 1.0
RO = 0.99
epsilon = 0.5
teacher_forcing = 0.5

data = DataPipelineRQL(batch_size=BATCH_SIZE)
en_vocab = data.en_vocab
spa_vocab = data.spa_vocab
train_loader = data.train_loader
valid_loader = data.valid_loader
test_loader = data.test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net(en_vocab, spa_vocab, RNN_HID_DIM, DROPOUT, NUM_RNN_LAYERS).to(device)
model = RQL(net, device, 64).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.999, last_epoch=-1)

mistranslation_loss = nn.CrossEntropyLoss(ignore_index=spa_vocab.stoi['<pad>'])
mistranslation_loss_per_agent = nn.CrossEntropyLoss(ignore_index=spa_vocab.stoi['<pad>'], reduction='none')
policy_loss = nn.MSELoss()

SPA_NULL = torch.tensor([spa_vocab.stoi['<null>']]).to(device)
EN_NULL = torch.tensor([en_vocab.stoi['<null>']]).to(device)
EN_PAD = torch.tensor([en_vocab.stoi['<pad>']]).to(device)
SPA_EOS = torch.tensor([spa_vocab.stoi['<eos>']]).to(device)
EN_EOS = torch.tensor([en_vocab.stoi['<eos>']]).to(device)

print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

_mistranslation_loss_weight, _policy_loss_weight = 0, 0
ro_to_k = 1
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_actions, ro_to_k, _mistranslation_loss_weight, _policy_loss_weight = train_epoch(epsilon, teacher_forcing, ro_to_k, _mistranslation_loss_weight, _policy_loss_weight)
    val_loss, val_bleu, val_actions = evaluate_epoch(valid_loader)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print('Train loss: {}, PPL: {}, epsilon: {}, teacher forcing: {}, action ratio: {}'.format(round(train_loss, 5), round(math.exp(train_loss), 3),  round(epsilon, 2),  round(teacher_forcing, 2), actions_ratio(train_actions)))
    print('Valid loss: {}, PPL: {}, BLEU: {}, action ratio: {}\n'.format(round(val_loss, 5), round(math.exp(val_loss), 3), round(100*val_bleu, 2), actions_ratio(val_actions)))

    lr_scheduler.step()
    epsilon = max(0.1, epsilon - 0.05)
    teacher_forcing = max(0.1, teacher_forcing - 0.00)

test_loss, test_bleu, test_actions = evaluate_epoch(test_loader)
print('Test loss: {}, PPL: {}, BLEU: {}, action ratio: {}\n'.format(round(test_loss, 5), round(math.exp(test_loss), 3), round(100*test_bleu, 2), actions_ratio(test_actions)))

