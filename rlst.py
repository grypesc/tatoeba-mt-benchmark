import argparse
import math
import os
import random
import time
import torch

import torch.optim as optim

from utils.data_pipeline import DataPipeline
from utils.tools import epoch_time, actions_ratio, save_model, BleuScorer, parse_utils
from criterions.rlst_criterion import RLSTCriterion
from models.rlst.rlst_fast import RLST, LeakyResidualApproximator, LeakyResidualNormApproximator

torch.set_printoptions(threshold=10_000)
random.seed(20)
torch.manual_seed(20)


def train_epoch(optimizer, epsilon, teacher_forcing, clip):
    model.train()
    rlst_criterion.train()
    epoch_mistranslation_loss = 0
    epoch_policy_loss = 0
    policy_multiplier = None
    total_actions = torch.zeros((3, 1), dtype=torch.long, device=device)
    for iteration, (src, trg) in enumerate(train_loader, 1):
        src, trg = src.T.to(device), trg.T.to(device)
        word_outputs, Q_used, Q_target, actions = model(src, trg, epsilon, teacher_forcing)
        total_actions += actions.cumsum(dim=1)
        optimizer.zero_grad()
        word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
        trg = trg.reshape(-1)

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
            src, trg = src.T.to(device), trg.T.to(device)
            word_outputs, _, _, actions = model(src)
            total_actions += actions.cumsum(dim=1)
            bleu_scorer.register_minibatch(word_outputs.permute(1, 0, 2), trg.T)
            word_outputs_clipped = word_outputs[:, :trg.size()[1], :]
            word_outputs_clipped = word_outputs_clipped.reshape(-1, word_outputs_clipped.shape[-1])
            trg = trg.reshape(-1)
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
                        default=0.2)
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

    net = LeakyResidualApproximator(src_vocab, trg_vocab, args.use_pretrained_embeddings, args.rnn_hid_dim, args.rnn_dropout, args.rnn_num_layers,
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

