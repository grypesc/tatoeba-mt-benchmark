import math
import random
import time
import torch

import torch.nn as nn
import torch.optim as optim

from utils.data_pipeline_rql import DataPipelineRQL
from utils.tools import epoch_time, bleu

torch.set_printoptions(threshold=10_000)
random.seed(20)
torch.manual_seed(20)


class RQL(nn.Module):
    def __init__(self,
                 input_dim: int,
                 src_emb_dim: int,
                 trg_emb_dim: int,
                 rnn_hid_dim: int,
                 num_layers: int,
                 output_dim: int):
        super().__init__()

        self.src_embedding = nn.Embedding(input_dim, src_emb_dim).from_pretrained(en_vocab.vectors, freeze=True)
        self.trg_embedding = nn.Embedding(input_dim, trg_emb_dim).from_pretrained(spa_vocab.vectors, freeze=True)
        self.rnn = nn.GRU(src_emb_dim + trg_emb_dim, rnn_hid_dim, num_layers=num_layers, bidirectional=False, dropout=0)
        self.output = nn.Linear(rnn_hid_dim, output_dim)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.src_embedding(src)
        trg_embedded = self.trg_embedding(previous_output)
        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        outputs = self.output(rnn_output)
        return outputs, rnn_state


BATCH_SIZE = 64
RNN_HID_DIM = 1024
NUM_LAYERS = 1
DISCOUNT = 0.99
epsilon = 0.5
teacher_forcing = 0.5
policy_loss_weight = 0.01

data = DataPipelineRQL(batch_size=BATCH_SIZE)
en_vocab = data.en_vocab
spa_vocab = data.spa_vocab
train_loader = data.train_loader
valid_loader = data.valid_loader
test_loader = data.test_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RQL(len(en_vocab), en_vocab.vectors.size()[1], spa_vocab.vectors.size()[1], RNN_HID_DIM, NUM_LAYERS, len(spa_vocab) + 3).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.999, last_epoch=-1)

word_loss = nn.CrossEntropyLoss(ignore_index=spa_vocab.stoi['<pad>'])
word_loss_per_agent = nn.CrossEntropyLoss(ignore_index=spa_vocab.stoi['<pad>'], reduction='none')
policy_loss = nn.MSELoss()

SPA_NULL = torch.tensor([spa_vocab.stoi['<null>']]).to(device)
EN_NULL = torch.tensor([en_vocab.stoi['<null>']]).to(device)
SPA_EOS = torch.tensor([spa_vocab.stoi['<eos>']]).to(device)
EN_EOS = torch.tensor([en_vocab.stoi['<eos>']]).to(device)


def episode(src, trg, epsilon, teacher_forcing):
    batch_size = src.size()[1]
    src_seq_len = src.size()[0]
    trg_seq_len = trg.size()[0]
    word_output = torch.tensor(([batch_size * [spa_vocab.stoi["<null>"]]]), device=device)
    rnn_state = torch.zeros((NUM_LAYERS, batch_size, RNN_HID_DIM), device=device)

    word_outputs = torch.zeros((trg_seq_len, batch_size, len(spa_vocab)), device=device)
    Q_used = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)
    Q_target = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)

    writing_agents = torch.full((batch_size,), False, device=device)
    terminated_agents = torch.full((batch_size,), False, device=device)
    terminated_on = torch.full((batch_size,), -1, device=device)

    i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)  # input indices
    j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)  # output indices
    t = 0  # time

    while True:
        input = torch.gather(src, 0, i)
        input[:, writing_agents] = EN_NULL
        output, rnn_state = model(input, word_output, rnn_state)
        _, word_output = torch.max(output[:, :, :-3], dim=2)

        if random.random() < epsilon:
            action = torch.randint(low=0, high=3, size=(1, batch_size), device=device)[0]
        else:
            action = torch.max(output[:, :, -3:], 2)[1][0]

        Q_used[t, :] = torch.gather(output[0, :, -3:], 1, action.unsqueeze(dim=1)).squeeze()
        Q_used[t, terminated_agents] = 0
        reading_agents = ~terminated_agents * (action == 0)
        writing_agents = ~terminated_agents * (action == 1)
        bothing_agents = ~terminated_agents * (action == 2)

        agents_outputting = writing_agents + bothing_agents
        word_outputs[j[:, agents_outputting], agents_outputting, :] = output[0, agents_outputting, :-3]

        just_terminated_agents = agents_outputting * (torch.gather(trg, 0, j) == SPA_EOS).squeeze_()
        naughty_agents = (reading_agents + bothing_agents) * (torch.gather(src, 0, i) == EN_EOS).squeeze_()
        just_terminated_agents = just_terminated_agents + naughty_agents
        i = i + (reading_agents + bothing_agents)
        old_j = j
        j = j + agents_outputting

        terminated_agents = terminated_agents + just_terminated_agents
        terminated_on[just_terminated_agents] = t

        i[i >= src_seq_len] = src_seq_len - 1
        j[j >= trg_seq_len] = trg_seq_len - 1

        if random.random() < teacher_forcing:
            word_output[:, :] = torch.gather(trg, 0, old_j)
        word_output[:, reading_agents] = SPA_NULL

        with torch.no_grad():
            _input = torch.gather(src, 0, i)
            _input[:, writing_agents] = EN_NULL
            _output, _ = model(_input, word_output, rnn_state)
            next_best_action_value, _ = torch.max(_output[:, :, -3:], 2)

            reward = (-1) * word_loss_per_agent(output[0, :, :-3], torch.gather(trg, 0, old_j)[0, :])
            Q_target[t, :] = reward + DISCOUNT * next_best_action_value
            Q_target[t, reading_agents] = next_best_action_value[0, reading_agents]
            Q_target[t, just_terminated_agents] = reward[just_terminated_agents]
            Q_target[t, naughty_agents] = -50.0

            if terminated_agents.all():
                return word_outputs, Q_used, Q_target
        t += 1


def train(epsilon, teacher_forcing, policy_loss_weight):
    model.train()
    epoch_loss = 0
    for iteration, (src, trg) in enumerate(train_loader):  # TODO: loss norm
        src, trg = src.to(device), trg.to(device)
        word_outputs, Q_used, Q_target = episode(src, trg, epsilon, teacher_forcing)
        optimizer.zero_grad()
        word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
        trg = trg.view(-1)
        _word_loss = word_loss(word_outputs, trg)
        loss = _word_loss + policy_loss_weight * policy_loss(Q_used, Q_target)
        loss.backward()
        optimizer.step()
        epoch_loss += _word_loss.item()
    return epoch_loss / len(train_loader)


def evaluate(loader):
    model.eval()
    epoch_loss, epoch_bleu = 0, 0
    with torch.no_grad():
        for iteration, (src, trg) in enumerate(loader):
            src, trg = src.to(device), trg.to(device)
            word_outputs, _, _ = episode(src, trg, 0, 0)
            epoch_bleu += bleu(word_outputs, trg, spa_vocab, device)
            word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
            trg = trg.view(-1)
            epoch_loss += word_loss(word_outputs, trg).item()
    return epoch_loss / len(loader), epoch_bleu / len(loader)


N_EPOCHS = 30

print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
# profile = cProfile.Profile()
# profile.enable()
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(epsilon, teacher_forcing, policy_loss_weight)
    val_loss, val_bleu = evaluate(valid_loader)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print('Train loss: {}, PPL: {}, epsilon: {}, teacher forcing: {}'.format(round(train_loss, 5), round(math.exp(train_loss), 3),  round(epsilon, 2),  round(teacher_forcing, 2)))
    print('Valid loss: {}, PPL: {}, BLEU: {}\n'.format(round(val_loss, 5), round(math.exp(val_loss), 3), round(100*val_bleu, 2)))

    lr_scheduler.step()
    epsilon = max(0.1, epsilon - 0.03)
    teacher_forcing = max(0.1, teacher_forcing - 0.0)

test_loss, test_bleu = evaluate(test_loader)
print('Test loss: {}, PPL: {}, BLEU: {}\n'.format(round(test_loss, 5), round(math.exp(test_loss), 3), round(100*test_bleu, 2)))

# profile.disable()
# profile.print_stats(sort='time')
