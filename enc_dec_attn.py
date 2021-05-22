import argparse
import math
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

from utils.data_pipeline import DataPipeline
from utils.tools import epoch_time, BleuScorer, save_model, parse_utils

random.seed(20)
torch.manual_seed(20)


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float,
                 use_pretrained_embeddings=False):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        if use_pretrained_embeddings:
            self.emb_dim = src_vocab.vectors.shape[1]
            self.embedding = nn.Embedding(input_dim, self.emb_dim).from_pretrained(src_vocab.vectors, freeze=True)
        self.rnn = nn.GRU(self.emb_dim, enc_hid_dim, num_layers=1, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
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
                 attention: nn.Module,
                 use_pretrained_embeddings=False,
                 decoder_dropout=0.0):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        if use_pretrained_embeddings:
            self.emb_dim = trg_vocab.vectors.shape[1]
            self.embedding = nn.Embedding(output_dim, self.emb_dim).from_pretrained(trg_vocab.vectors, freeze=True)
        self.rnn = nn.GRU((enc_hid_dim * 2) + self.emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + self.emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder_dropout = nn.Dropout(decoder_dropout)

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
                encoder_outputs: Tensor):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output_in = torch.cat((output, weighted_encoder_rep, embedded), dim=1)
        output_in = self.decoder_dropout(output_in)
        output = self.out(output_in)

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

    def forward(self, src, trg, teacher_forcing_ratio):
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


def train(model, data_loader, optimizer, criterion, clip, teacher_forcing):
    model.train()
    epoch_loss = 0

    for _, (src, trg) in enumerate(data_loader):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing)
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
            output = model(src, trg, 0.0)  # turn off teacher forcing
            bleu_scorer.register_minibatch(output[1:, :, :], trg[1:, :])
            output_clipped = output[:trg.size()[0], :, :]
            output_clipped = output_clipped[1:].view(-1, output_clipped.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output_clipped, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader), bleu_scorer.epoch_score()


def parse_args():
    parser = argparse.ArgumentParser()
    parse_utils(parser)
    parser.add_argument('--test-seq-max-len',
                        help='maximum length of sequence that can be produced during testing',
                        type=int,
                        default=64)
    parser.add_argument('--teacher-forcing',
                        help='teacher forcing',
                        type=float,
                        default=1.0)
    parser.add_argument('--enc-hid-dim',
                        help='encoder hidden size',
                        type=int,
                        default=128)
    parser.add_argument('--dec-hid-dim',
                        help='decoder hidden size',
                        type=int,
                        default=128)
    parser.add_argument('--attn-dim',
                        help='attention layer size',
                        type=int,
                        default=32)
    parser.add_argument('--decoder-dropout',
                        help='dropout before last output layer',
                        type=float,
                        default=0.0)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    data = DataPipeline(batch_size=args.batch_size, src_lang=args.src, trg_lang=args.trg,
                        token_min_freq=args.token_min_freq, use_pretrained_embeds=args.use_pretrained_embeddings)
    src_vocab = data.src_vocab
    trg_vocab = data.trg_vocab
    train_loader = data.train_loader
    valid_loader = data.valid_loader
    test_loader = data.test_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = Encoder(len(src_vocab), args.src_embed_dim, args.enc_hid_dim, args.dec_hid_dim, args.embed_dropout, args.use_pretrained_embeddings)
    attn = Attention(args.enc_hid_dim, args.dec_hid_dim, args.attn_dim)
    dec = Decoder(len(trg_vocab), args.trg_embed_dim, args.enc_hid_dim, args.dec_hid_dim, args.embed_dropout, attn, args.use_pretrained_embeddings, args.decoder_dropout)
    model = Seq2Seq(enc, dec, device, args.test_seq_max_len).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi['<pad>'])

    if args.load_model_name:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.load_model_name)))

    print(vars(args))
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters\n')

    bleu_scorer = BleuScorer(trg_vocab, device)
    best_val_bleu = 0.0

    if not args.test:
        for epoch in range(args.epochs):
            start_time = time.time()
            train_loss = train(model, train_loader, optimizer, criterion, args.clip, args.teacher_forcing)
            valid_loss, valid_bleu = evaluate(model, valid_loader, criterion, bleu_scorer)

            save_model(model, args.checkpoint_dir, "enc_dec_attn", valid_bleu > best_val_bleu)
            best_val_bleu = valid_bleu if valid_bleu > best_val_bleu else best_val_bleu

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'Train loss: {train_loss:.3f}, PPL: {math.exp(train_loss):7.3f}')
            print(f'Valid loss: {valid_loss:.3f}, PPL: {math.exp(valid_loss):7.3f}, BLEU: {round(100*valid_bleu, 2)}\n')

    else:
        test_loss, test_bleu = evaluate(model, test_loader, criterion, bleu_scorer)
        print(f'Test loss: {test_loss:.3f}, PPL: {math.exp(test_loss):7.3f}, BLEU: {round(100*test_bleu, 2)}')
        test_loss, test_bleu = evaluate(model, data.long_test_loader, criterion, bleu_scorer)
        print(f'Long test loss: {test_loss:.3f}, PPL: {math.exp(test_loss):7.3f}, BLEU: {round(100*test_bleu, 2)}')
