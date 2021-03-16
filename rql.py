import math
import random
import time
import torch

import torch.nn as nn
import torch.optim as optim

from utils.data_pipeline_rql import DataPipelineRQL
from utils.tools import epoch_time, bleu, actions_ratio
from utils.rql_models import RQL, RQL1, RQL2

torch.set_printoptions(threshold=10_000)
random.seed(20)
torch.manual_seed(20)


def episode(src, trg, epsilon, teacher_forcing):
    batch_size = src.size()[1]
    src_seq_len = src.size()[0]
    trg_seq_len = trg.size()[0]
    word_output = torch.tensor(([batch_size * [spa_vocab.stoi["<null>"]]]), device=device)
    rnn_state = torch.zeros((NUM_RNN_LAYERS, batch_size, RNN_HID_DIM), device=device)

    word_outputs = torch.zeros((trg_seq_len, batch_size, len(spa_vocab)), device=device)
    Q_used = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)
    Q_target = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)

    writing_agents = torch.full((batch_size,), False, device=device)
    naughty_agents = torch.full((batch_size,), False, device=device)  # Want more input after input eos
    terminated_agents = torch.full((batch_size,), False, device=device)
    # terminated_on = torch.full((batch_size,), -1, device=device)

    i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)  # input indices
    j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)  # output indices
    t = 0  # time
    actions_count = torch.zeros(3, dtype=torch.long, device=device)

    while True:
        input = torch.gather(src, 0, i)
        input[:, writing_agents] = EN_NULL
        input[:, naughty_agents] = EN_PAD
        output, rnn_state = model(input, word_output, rnn_state)
        _, word_output = torch.max(output[:, :, :-3], dim=2)

        if random.random() < epsilon:
            action = torch.randint(low=0, high=3, size=(1, batch_size), device=device)[0]
        else:
            action = torch.max(output[:, :, -3:], 2)[1][0]

        Q_used[t, :] = torch.gather(output[0, :, -3:], 1, action.unsqueeze(dim=1)).squeeze()
        Q_used[t, terminated_agents] = 0

        with torch.no_grad():
            reading_agents = ~terminated_agents * (action == 0)
            writing_agents = ~terminated_agents * (action == 1)
            bothing_agents = ~terminated_agents * (action == 2)

            actions_count[0] += reading_agents.sum()
            actions_count[1] += writing_agents.sum()
            actions_count[2] += bothing_agents.sum()

        agents_outputting = writing_agents + bothing_agents
        word_outputs[j[:, agents_outputting], agents_outputting, :] = output[0, agents_outputting, :-3]

        just_terminated_agents = agents_outputting * (torch.gather(trg, 0, j) == SPA_EOS).squeeze_()
        naughty_agents = (reading_agents + bothing_agents) * (torch.gather(src, 0, i) == EN_EOS).squeeze_()
        i = i + ~naughty_agents * (reading_agents + bothing_agents)
        old_j = j
        j = j + agents_outputting

        terminated_agents = terminated_agents + just_terminated_agents
        # terminated_on[just_terminated_agents] = t

        i[i >= src_seq_len] = src_seq_len - 1
        j[j >= trg_seq_len] = trg_seq_len - 1

        if random.random() < teacher_forcing:
            word_output = torch.gather(trg, 0, old_j)
        word_output[:, reading_agents] = SPA_NULL

        with torch.no_grad():
            _input = torch.gather(src, 0, i)
            _input[:, writing_agents] = EN_NULL
            _input[:, naughty_agents] = EN_PAD
            _output, _ = model(_input, word_output, rnn_state)
            next_best_action_value, _ = torch.max(_output[:, :, -3:], 2)

            reward = (-1) * mistranslation_loss_per_agent(output[0, :, :-3], torch.gather(trg, 0, old_j)[0, :])
            Q_target[t, :] = reward + DISCOUNT * next_best_action_value
            Q_target[t, terminated_agents] = 0
            Q_target[t, reading_agents] = next_best_action_value[0, reading_agents]
            Q_target[t, reading_agents * naughty_agents] = DISCOUNT * next_best_action_value[0, reading_agents]
            Q_target[t, just_terminated_agents] = reward[just_terminated_agents]
            Q_target[t, naughty_agents] = Q_target[t, naughty_agents] - 10.0

            if terminated_agents.all() or t >= src_seq_len + trg_seq_len - 1:
                return word_outputs, Q_used, Q_target, actions_count
        t += 1


def train(epsilon, teacher_forcing):
    model.train()
    epoch_loss = 0
    loss_avg_decay = 0.99
    _mistranslation_loss_weight, _policy_loss_weight = 0, 0
    total_actions = torch.zeros(3, dtype=torch.long, device=device)
    for iteration, (src, trg) in enumerate(train_loader, 1):
        w_k = loss_avg_decay * (1 - loss_avg_decay ** (iteration - 1)) / (1 - loss_avg_decay ** iteration)
        src, trg = src.to(device), trg.to(device)
        word_outputs, Q_used, Q_target, actions = episode(src, trg, epsilon, teacher_forcing)
        total_actions += actions
        optimizer.zero_grad()
        word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
        trg = trg.view(-1)

        _policy_loss = policy_loss(Q_used, Q_target)
        _mistranslation_loss = mistranslation_loss(word_outputs, trg)

        _mistranslation_loss_weight = w_k * _mistranslation_loss_weight + (1 - w_k) * float(_mistranslation_loss)
        _policy_loss_weight = w_k * _policy_loss_weight + (1 - w_k) * float(_policy_loss)
        loss = _policy_loss / _policy_loss_weight + MISTRANSLATION_LOSS_MULTIPLIER * _mistranslation_loss / _mistranslation_loss_weight

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        epoch_loss += _mistranslation_loss.item()
    return epoch_loss / len(train_loader), total_actions.tolist()


def evaluate(loader):
    model.eval()
    epoch_loss, epoch_bleu = 0, 0
    total_actions = torch.zeros(3, dtype=torch.long, device=device)
    with torch.no_grad():
        for iteration, (src, trg) in enumerate(loader):
            src, trg = src.to(device), trg.to(device)
            word_outputs, _, _, actions = episode(src, trg, 0, 0)
            total_actions += actions
            epoch_bleu += bleu(word_outputs, trg, spa_vocab, SPA_EOS)
            word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
            trg = trg.view(-1)
            epoch_loss += mistranslation_loss(word_outputs, trg).item()
    return epoch_loss / len(loader), epoch_bleu / len(loader), total_actions.tolist()


BATCH_SIZE = 64
RNN_HID_DIM = 256
DROPOUT = 0.0
NUM_RNN_LAYERS = 1
DISCOUNT = 0.99
MISTRANSLATION_LOSS_MULTIPLIER = 30
epsilon = 0.5
teacher_forcing = 0.5


data = DataPipelineRQL(batch_size=BATCH_SIZE)
en_vocab = data.en_vocab
spa_vocab = data.spa_vocab
train_loader = data.train_loader
valid_loader = data.valid_loader
test_loader = data.test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = nn.DataParallel(RQL(en_vocab, spa_vocab, RNN_HID_DIM, DROPOUT, NUM_RNN_LAYERS).to(device))

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

N_EPOCHS = 30

print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
# profile = cProfile.Profile()
# profile.enable()
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_actions = train(epsilon, teacher_forcing)
    val_loss, val_bleu, val_actions = evaluate(valid_loader)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print('Train loss: {}, PPL: {}, epsilon: {}, teacher forcing: {}, action ratio: {}'.format(round(train_loss, 5), round(math.exp(train_loss), 3),  round(epsilon, 2),  round(teacher_forcing, 2), actions_ratio(train_actions)))
    print('Valid loss: {}, PPL: {}, BLEU: {}, action ratio: {}\n'.format(round(val_loss, 5), round(math.exp(val_loss), 3), round(100*val_bleu, 2), actions_ratio(val_actions)))

    lr_scheduler.step()
    epsilon = max(0.1, epsilon - 0.05)
    teacher_forcing = max(0.1, teacher_forcing - 0.05)

test_loss, test_bleu, test_actions = evaluate(test_loader)
print('Test loss: {}, PPL: {}, BLEU: {}, action ratio: {}\n'.format(round(test_loss, 5), round(math.exp(test_loss), 3), round(100*test_bleu, 2), actions_ratio(test_actions)))

# profile.disable()
# profile.print_stats(sort='time')
