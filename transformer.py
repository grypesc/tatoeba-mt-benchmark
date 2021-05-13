import argparse
import math
import os
import random
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

from transformer_model.transformer import Transformer
from utils.data_pipeline import DataPipeline
from utils.tools import BleuScorer, epoch_time, parse_utils, save_model

torch.set_printoptions(threshold=10_000)
random.seed(12)
torch.manual_seed(12)


def make_mask(src_input, trg_input, pad_id):
    seq_len = trg_input.shape[-1]
    e_mask = (src_input != pad_id).unsqueeze(1)  # (B, 1, L)
    d_mask = (trg_input != pad_id).unsqueeze(1)  # (B, 1, L)

    nopeak_mask = torch.ones([1, seq_len, seq_len], dtype=torch.bool)  # (1, L, L)
    nopeak_mask = torch.tril(nopeak_mask).to(device)  # (1, L, L) to triangular shape

    d_mask = d_mask & nopeak_mask  # (B, L, L) padding false
    return e_mask, d_mask


def train(model, data_loader, optimizer, criterion, clip, scheduler):
    model.train()
    epoch_loss = 0

    for _, (src, trg) in enumerate(data_loader):
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)
        scheduler.adjust_optim()
        dec_inp = trg[:, :-1]
        trg = trg[:, 1:]
        src, trg, dec_inp = src.to(device), trg.to(device), dec_inp.to(device)
        e_mask, d_mask = make_mask(src, dec_inp, SRC_PAD)
        optimizer.zero_grad()
        output = model(src, dec_inp, e_mask, d_mask, training=True)
        output = output.view(-1, output.shape[-1])
        trg = torch.reshape(trg, (-1,))
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
            src = src.permute(1, 0)
            trg = trg.permute(1, 0)
            dec_inp = trg[:, :1]
            trg = trg[:, 1:]
            src, trg, dec_inp = src.to(device), trg.to(device), dec_inp.to(device)
            e_mask, d_mask = make_mask(src, dec_inp, SRC_PAD)
            output = model(src, dec_inp, e_mask, d_mask, training=False, limit=trg.shape[-1])
            bleu_scorer.register_minibatch(output.permute(1, 0, 2), trg.permute(1, 0))
            output = output.view(-1, output.shape[-1])
            trg = torch.reshape(trg, (-1,))
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader), bleu_scorer.epoch_score()


class Adjuster:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.n_iter = 1

    def adjust_optim(self):
        arg1 = 1 / math.sqrt(self.n_iter)
        arg2 = self.n_iter * (self.warmup_steps ** -1.5)

        self.optimizer.param_groups[0]["lr"] = (1 / self.d_model) * min([arg1, arg2])
        self.n_iter += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parse_utils(parser)
    parser.add_argument("--warmup_steps", help="Defines warmup steps during training", type=int, default=4000)
    parser.add_argument("--lr", help="Defines initial learning rate", type=float, default=0.0001)
    parser.add_argument("--d_model", help="Transformer model d_model param", type=int, default=512)
    parser.add_argument("--num_heads", help="Transformer model atention heads number", type=int, default=8)
    parser.add_argument("--num_layers", help="Transformer model enc/dec stacks layers", type=int, default=6)
    parser.add_argument("--d_ffn", help="Transformer model ffn network internal dimension", type=int, default=2048)
    parser.add_argument("--dropout", help="Dropout rate", type=float, default=0.1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    BATCH_SIZE = args.batch_size
    if args.use_pretrained_embeddings:
        D_MODEL = 303
        NHEAD = 3
    else:
        D_MODEL = args.d_model
        NHEAD = args.num_heads
    NUM_LAYERS = args.num_layers
    DIM_FEEDFORWARD = args.d_ffn
    DROPOUT = args.dropout
    CLIP = args.clip
    MAX_LEN = 100
    N_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    WARMUP_STEPS = args.warmup_steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parameters = {
        "device": device,
        "max_seq_len": MAX_LEN,
        "num_heads": NHEAD,
        "num_layers": NUM_LAYERS,
        "d_model": D_MODEL,
        "d_ff": DIM_FEEDFORWARD,
        "d_k": D_MODEL // NHEAD,
        "drop_out_rate": DROPOUT,
    }

    data = DataPipeline(
        batch_size=BATCH_SIZE,
        src_lang=args.src,
        trg_lang=args.trg,
        token_min_freq=args.token_min_freq,
        use_pretrained_embeds=args.use_pretrained_embeddings,
    )
    src_vocab = data.src_vocab
    trg_vocab = data.trg_vocab
    train_loader = data.train_loader
    valid_loader = data.valid_loader
    test_loader = data.test_loader

    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)
    SRC_PAD = src_vocab.stoi["<pad>"]
    TRG_PAD = trg_vocab.stoi["<pad>"]
    TRG_BOS = trg_vocab.stoi["<bos>"]
    src_embeddings, trg_embeddings = None, None
    if args.use_pretrained_embeddings:
        src_embeddings = src_vocab.vectors
        trg_embeddings = trg_vocab.vectors

    assert SRC_PAD == TRG_PAD

    parameters["src_vocab_size"] = INPUT_DIM
    parameters["trg_vocab_size"] = OUTPUT_DIM
    parameters["src_embedding"] = src_embeddings
    parameters["trg_embedding"] = trg_embeddings

    model = Transformer(parameters).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98))
    scheduler = Adjuster(optimizer, D_MODEL, WARMUP_STEPS)
    criterion = nn.NLLLoss(ignore_index=trg_vocab.stoi["<pad>"])

    if args.load_model_name:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.load_model_name)))

    bleu_scorer = BleuScorer(trg_vocab, device)
    best_val_bleu = 0.0

    print(vars(args))
    print(f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    if not args.test:
        for epoch in range(N_EPOCHS):
            start_time = time.time()
            train_loss = train(model, train_loader, optimizer, criterion, CLIP, scheduler)
            valid_loss, valid_bleu = evaluate(model, valid_loader, criterion, bleu_scorer)

            save_model(model, args.checkpoint_dir, "transformer", valid_bleu > best_val_bleu)
            best_val_bleu = valid_bleu if valid_bleu > best_val_bleu else best_val_bleu

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"Train loss: {train_loss:.3f}, PPL: {math.exp(train_loss):7.3f}")
            print(f"Valid loss: {valid_loss:.3f}, PPL: {math.exp(valid_loss):7.3f}, BLEU: {round(100*valid_bleu, 2)}\n")

    else:
        test_loss, test_bleu = evaluate(model, test_loader, criterion, bleu_scorer)
        print(f"Test loss: {test_loss:.3f}, PPL: {math.exp(test_loss):7.3f}, BLEU: {round(100*test_bleu, 2)}")
        test_loss, test_bleu = evaluate(model, data.long_test_loader, criterion, bleu_scorer)
        print(f"Long test loss: {test_loss:.3f}, PPL: {math.exp(test_loss):7.3f}, BLEU: {round(100*test_bleu, 2)}")