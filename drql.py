import random
import torch

import torch.nn as nn
import torch.optim as optim

from utils.data_pipeline_drql import DataPipeline


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
        rnn_input = torch.unsqueeze(torch.cat((src_embedded, trg_embedded), dim=1), dim=0)
        rnn_output, rnn_state = self.rnn(rnn_input, rnn_state)
        outputs = self.output(rnn_output)
        return outputs, rnn_state


data = DataPipeline(batch_size=1)
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

optimizer = optim.SGD(model.parameters(), lr=1e-3)
word_loss = nn.CrossEntropyLoss()
policy_loss = nn.MSELoss()
sentence_loss = nn.CrossEntropyLoss()


def train(epsilon):
    model.train()
    epoch_sentence_loss = 0
    for iteration, (src, trg) in enumerate(train_loader):
        src, trg = src.to(device), trg.to(device)
        src_seq_len = src.size()[0]
        trg_seq_len = trg.size()[0]

        i, j = 0, 0
        is_terminal_state = False
        input_word = src[0, :]
        word_output = torch.tensor([spa_vocab.stoi['<null>']]).to(device)
        rnn_state = torch.zeros((1, 1, RNN_HID_DIM)).to(device)
        trg_word = trg[0, :]
        output_sentence = []

        while not is_terminal_state:
            _word_loss = None
            optimizer.zero_grad()
            output, rnn_state = model(input_word, word_output, rnn_state)
            policy_output = output[0, 0, -3:]

            _, action_index = torch.max(policy_output, 0)  # Epsilon greedy strategy
            if random.random() < max(0.05, epsilon):
                action_index = random.randint(0, 2)

            if action_index == 0:  # WAIT
                word_output = torch.tensor([spa_vocab.stoi['<null>']]).to(device)  # not sure
                reward = -0.5
                i += 1

            elif action_index == 1:  # SKIP
                word_output = output[0, :, :-3]
                _word_loss = word_loss(word_output, trg_word)
                _, word_output = torch.max(word_output, dim=1)
                reward = -1.0
                if word_output == trg_word:
                    reward = 5.0
                j += 1
                output_sentence.append(output[0, 0, :-3])

            else:  # GO
                word_output = output[0, :, :-3]
                _word_loss = word_loss(word_output, trg_word)
                _, word_output = torch.max(word_output, dim=1)
                reward = -1.0
                if word_output == trg_word:
                    reward = 5.0
                i += 1
                j += 1
                output_sentence.append(output[0, 0, :-3])
            old_trg_word = trg_word
            if action_index == 1:  # SKIP
                input_word = torch.tensor([en_vocab.stoi['<null>']]).to(device)
            elif i < src_seq_len:
                input_word = src[i, :]
            else:
                input_word = torch.tensor([en_vocab.stoi['<eos>']]).to(device)
                reward = -2.0
            if action_index == 0:  # WAIT
                pass
            elif j < trg_seq_len:
                trg_word = trg[j, :]
            else:
                trg_word = torch.tensor([spa_vocab.stoi['<eos>']]).to(device)
                reward = -2.0

            with torch.no_grad():  # Forward pass to get the next action
                _output, _ = model(input_word, word_output, rnn_state)
                _policy_output = _output[0, 0, -3:]
                _, best_action_index = torch.max(_policy_output, 0)

            if word_output[0] == spa_vocab.stoi["<eos>"]:  # Terminal state on eos
                _policy_loss = policy_loss(policy_output[action_index], torch.tensor(reward).to(device))
                is_terminal_state = True
            elif i > 2*src_seq_len or j > 2*trg_seq_len:  # Terminal state when agent is too slow
                is_terminal_state = True
                reward = -10.0
                _policy_loss = policy_loss(policy_output[action_index], torch.tensor(reward).to(device))
            else:
                _policy_loss = policy_loss(policy_output[action_index],
                                           reward + DISCOUNT * _policy_output[best_action_index])

            if action_index == 0:
                _policy_loss.backward()
            else:
                (_policy_loss + _word_loss).backward()
            optimizer.step()
            rnn_state.detach_()
            output.detach_()
            if random.random() < 0.5 and action_index != 0:  # Teacher forcing
                word_output = old_trg_word

        if output_sentence:
            predicted_sentence = torch.stack(output_sentence)
        else:
            predicted_sentence = torch.zeros(1, len(spa_vocab)).to(device)

        trg_t = torch.squeeze(trg)
        out_len_diff = predicted_sentence.size()[0] - trg_t.size()[0]
        if out_len_diff > 0:
            trg_t = torch.cat((trg_t, *out_len_diff * [torch.tensor([spa_vocab.stoi['<unk>']]).to(device)]))
        elif out_len_diff < 0:
            predicted_sentence = torch.cat(
                (predicted_sentence, *(-1) * out_len_diff * [torch.zeros(1, len(spa_vocab)).to(device)]))
        epoch_sentence_loss += sentence_loss(predicted_sentence, trg_t).item()
        epsilon -= 0.0001
        # print(iteration, action_index)
    return epoch_sentence_loss / len(train_loader), epsilon


N_EPOCHS = 100

print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

for epoch in range(N_EPOCHS):
    train_loss, epsilon = train(epsilon)
    print(train_loss)
