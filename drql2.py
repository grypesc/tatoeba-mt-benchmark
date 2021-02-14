import random
import time
import torch

import torch.nn as nn
import torch.optim as optim

from utils.data_pipeline_drql2 import DataPipeline
from utils.tools import epoch_time, int_to_one_hot

torch.set_printoptions(threshold=10_000)

class DRQL(nn.Module):
    def __init__(self,
                 input_dim: int,
                 src_emb_dim: int,
                 trg_emb_dim: int,
                 rnn_hid_dim: int,
                 output_dim: int):
        super().__init__()

        self.src_embedding = nn.Embedding(input_dim, src_emb_dim).from_pretrained(en_vocab.vectors, freeze=True)
        self.trg_embedding = nn.Embedding(input_dim, trg_emb_dim).from_pretrained(spa_vocab.vectors, freeze=True)
        self.rnn = nn.GRU(src_emb_dim + trg_emb_dim, rnn_hid_dim, num_layers=1, bidirectional=False)
        self.output = nn.Linear(rnn_hid_dim, output_dim)

    def forward(self, src, previous_output, rnn_state):
        src_embedded = self.src_embedding(src)
        trg_embedded = self.trg_embedding(previous_output)
        rnn_input = torch.cat((src_embedded, trg_embedded), dim=2)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        outputs = self.output(rnn_output)
        return outputs, rnn_state

BATCH_SIZE = 128
data = DataPipeline(batch_size=BATCH_SIZE)
en_vocab = data.en_vocab
spa_vocab = data.spa_vocab
train_loader = data.train_loader
valid_loader = data.valid_loader
test_loader = data.test_loader

INPUT_DIM = len(en_vocab)
OUTPUT_DIM = len(spa_vocab) + 3

SRC_EMB_DIM = en_vocab.vectors.size()[1]
TRG_EMB_DIM = spa_vocab.vectors.size()[1]
RNN_HID_DIM = 512
DISCOUNT = 0.99
epsilon = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DRQL(INPUT_DIM, SRC_EMB_DIM, TRG_EMB_DIM, RNN_HID_DIM, OUTPUT_DIM).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99, last_epoch=-1)

word_loss = nn.CrossEntropyLoss(ignore_index=spa_vocab.stoi['<pad>'])
policy_loss = nn.MSELoss()

SPA_NULL = torch.tensor([spa_vocab.stoi['<null>']]).to(device)
EN_NULL = torch.tensor([en_vocab.stoi['<null>']]).to(device)
SPA_EOS = torch.tensor([spa_vocab.stoi['<eos>']]).to(device)
EN_EOS = torch.tensor([en_vocab.stoi['<eos>']]).to(device)
SPA_UNK = torch.tensor([spa_vocab.stoi['<unk>']]).to(device)

EN_NULL_one_hot = int_to_one_hot(en_vocab["<null>"], len(en_vocab), device)
SPA_SOS_ONE_HOT = int_to_one_hot(spa_vocab["<sos>"], len(spa_vocab), device)
SPA_NULL_ONE_HOT = int_to_one_hot(spa_vocab["<null>"], len(spa_vocab), device)


def train(epsilon):
    model.train()
    epoch_loss = 0
    for iteration, (src, trg) in enumerate(train_loader):
        src, trg = src.to(device), trg.to(device)
        batch_size = src.size()[1]  # current
        src_seq_len = src.size()[0]
        trg_seq_len = trg.size()[0]
        word_output = torch.tensor(([batch_size*[spa_vocab.stoi["<sos>"]]]), device=device)
        rnn_state = torch.zeros((1, batch_size, RNN_HID_DIM), device=device)

        word_outputs = torch.zeros((trg_seq_len, batch_size, len(spa_vocab)),  device=device)
        policy_outputs = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)
        target_policy = torch.zeros((src_seq_len + trg_seq_len, batch_size), device=device)

        skipping_agents = torch.full((batch_size, ), False, device=device)

        REWARD_PADDING = torch.full((1, batch_size), fill_value=True, device=device)
        i = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)
        j = torch.zeros(size=(1, batch_size), dtype=torch.long, device=device)
        t = 0
        is_terminal = False

        while not is_terminal:
            input = torch.gather(src, 0, i)
            input[:, skipping_agents] = EN_NULL
            output, rnn_state = model(input, word_output, rnn_state)
            _, word_output = torch.max(output[:, :, :-3], dim=2)
            policy_output = output[:, :, -3:]

            if random.random() < epsilon:
                action = torch.randint(low=0, high=3, size=(1, batch_size), device=device)[0]
            else:
                action = torch.max(policy_output, 2)[1][0].to(device)

            policy_outputs[t, :] = torch.gather(policy_output[0, :, :], 1, action.unsqueeze(dim=1)).squeeze()
            waiting_agents = (action == 0).to(device)
            skipping_agents = (action == 1).to(device)
            going_agents = (action == 2).to(device)

            word_outputs[j[:, skipping_agents + going_agents], skipping_agents + going_agents, :] = output[0, skipping_agents + going_agents, :-3]
            if random.random() < 0.5:  # Teacher forcing
                word_output[:, :] = torch.gather(trg, 0, j)
            word_output[:, waiting_agents] = SPA_NULL

            i = i + waiting_agents + going_agents
            j = j + skipping_agents + going_agents

            with torch.no_grad():
                if torch.max(i, dim=1)[0] >= src_seq_len or torch.max(j, dim=1)[0] >= trg_seq_len:
                    is_terminal = True
                    trg_is_eos = torch.eq(trg, SPA_EOS)
                    trg_eos = trg_is_eos.max(0)[1]
                    word_outputs_is_eos = torch.eq(torch.max(word_outputs, dim=2)[1], SPA_EOS)
                    word_outputs_is_eos[-1, :] = REWARD_PADDING
                    word_outputs_eos = word_outputs_is_eos.max(0)[1]
                    reward = (-1) * torch.abs(trg_eos - word_outputs_eos).float()
                    target_policy.scatter_(0, word_outputs_eos.unsqueeze(0), reward.unsqueeze(0))
                    # mask = torch.zeros_like(target_policy, device=device)
                    # mask = torch.scatter(mask, 0, word_outputs_eos.unsqueeze(0), 1)
                    # mask = mask.cumsum(dim=0)

                else:
                    _input = torch.gather(src, 0, i)
                    _input[:, skipping_agents] = EN_NULL
                    _output, _ = model(_input, word_output, rnn_state)
                    _policy_output = _output[:, :, -3:]
                    action_value, _ = torch.max(_policy_output, 2)
                    target_policy[t, :] = DISCOUNT * action_value
            t += 1

        optimizer.zero_grad()
        word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
        trg = trg.view(-1)
        _word_loss = word_loss(word_outputs, trg)
        loss = _word_loss + policy_loss(policy_outputs, target_policy)
        loss.backward()
        optimizer.step()
        epoch_loss += _word_loss.item()
    return epoch_loss / len(train_loader), epsilon


N_EPOCHS = 1000

print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
# profile = cProfile.Profile()
# profile.enable()
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, epsilon = train(epsilon)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    lr_scheduler.step()
    epsilon = max(0.05, epsilon - 0.002)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print('Train loss: {}, epsilon: {}'.format(round(train_loss, 10), round(epsilon, 2)))

# profile.disable()
# profile.print_stats(sort='time')
