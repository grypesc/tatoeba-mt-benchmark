import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

from utils.data_pipeline import DataPipeline
from utils.tools import epoch_time, BleuScorer

random.seed(20)
torch.manual_seed(20)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim).from_pretrained(en_vocab.vectors, freeze=True)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))

        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim).from_pretrained(spa_vocab.vectors, freeze=True)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim=1))

        return output, decoder_hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device,
                 max_len: int):

        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_len = max_len

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:
        batch_size = src.shape[1]

        max_len = self.max_len
        if self.training:
            max_len = trg.shape[0]

        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <bos> token
        output = trg[0, :]

        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


def train(model, data_loader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for _, (src, trg) in enumerate(data_loader):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def evaluate(model, data_loader, criterion, bleu_scorer):
    model.eval()
    epoch_loss, epoch_bleu = 0, 0
    with torch.no_grad():
        for _, (src, trg) in enumerate(data_loader):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing
            bleu_scorer.register_minibatch(output[1:, :, :], trg[1:, :])
            output_clipped = output[:trg.size()[0], :, :]
            output_clipped = output_clipped[1:].view(-1, output_clipped.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output_clipped, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader), bleu_scorer.epoch_score()


if __name__ == '__main__':

    BATCH_SIZE = 128
    N_EPOCHS = 30
    ENC_HID_DIM = 128
    DEC_HID_DIM = 128
    ATTN_DIM = 32
    ENC_DROPOUT = 0.0  # This comes after pretrained embedding layers so in theory it's better not to increase it
    DEC_DROPOUT = 0.0
    CLIP = 1
    TEST_SEQUENCE_MAX_LENGTH = 64

    data = DataPipeline(batch_size=BATCH_SIZE)
    en_vocab = data.en_vocab
    spa_vocab = data.spa_vocab
    train_loader = data.train_loader
    valid_loader = data.valid_loader
    test_loader = data.test_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(spa_vocab)

    enc = Encoder(INPUT_DIM, en_vocab.vectors.size()[1], ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, spa_vocab.vectors.size()[1], ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device, TEST_SEQUENCE_MAX_LENGTH).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=spa_vocab.stoi['<pad>'])

    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    bleu_scorer = BleuScorer(spa_vocab, device)
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss, valid_bleu = evaluate(model, valid_loader, criterion, bleu_scorer)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain loss: {train_loss:.3f}, PPL: {math.exp(train_loss):7.3f}')
        print(f'\tValid loss: {valid_loss:.3f}, PPL: {math.exp(valid_loss):7.3f}, BLEU: {round(100*valid_bleu, 2)}')

    test_loss, test_bleu = evaluate(model, test_loader, criterion, bleu_scorer)

    print(f'\tTest loss: {test_loss:.3f}, PPL: {math.exp(test_loss):7.3f}, BLEU: {round(100*test_bleu, 2)}')
