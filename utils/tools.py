import os
import torch

from pathlib import Path
from torchtext.data.metrics import bleu_score


class BleuScorer:
    """This class is responsible for a fair BLEU score calculation for the whole epoch using torchtext
    bleu implementation. Every inference minibatch you should register target tokens and output token probabilities.
    At the end of epoch call epoch_score() to get bleu score."""

    def __init__(self, trg_vocab, device):
        self.vocab = trg_vocab
        self.device = device
        self.output_str, self.target_str = [], []
        self.eos = torch.tensor(trg_vocab.stoi['<eos>']).to(device)

    def register_minibatch(self, output, target):
        """
        Registers minibatch tokens as strings, uses eos token to trim sequences
        :param output: Torch.tensor of shape (seq_len x batch_size x target_corpus_length)
        :param target: Torch.tensor of shape (seq_len x batch_size)
        """
        _, output_words = torch.max(output, 2)
        batch_size = output_words.size()[1]
        eos = torch.tensor(self.vocab.stoi['<eos>']).to(self.device)
        eos_vector = torch.full((1, batch_size), eos, device=self.device)

        output_words = torch.cat((output_words, eos_vector), dim=0)
        output_is_eos = torch.eq(output_words, eos)
        output_eos = output_is_eos.max(0)[1]
        output_words = output_words.transpose(1, 0)

        trg_is_eos = torch.eq(target, eos)
        trg_eos = trg_is_eos.max(0)[1]

        target = target.transpose(1, 0)
        target = [x[:trg_eos[i]] for i, x in enumerate(target)]
        target_str = [[[self.vocab.itos[x] for x in y]] for y in target]
        self.target_str.extend(target_str)

        output = [x[:output_eos[i]] for i, x in enumerate(output_words)]
        output_str = [[self.vocab.itos[x] for x in y] for y in output]
        self.output_str.extend(output_str)

    def epoch_score(self):
        bleu = bleu_score(self.output_str, self.target_str)
        self.output_str, self.target_str = [], []
        return bleu


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def actions_ratio(actions):
    s = sum(actions)
    a = [actions[0]/s, actions[1]/s, actions[2]/s]
    return [round(action, 2) for action in a]


def save_model(model, path, model_name, is_best):
    Path(path).mkdir(parents=True, exist_ok=True)
    if is_best:
        torch.save(model.state_dict(), os.path.join(path, model_name + "_best.pth"))
    torch.save(model.state_dict(), os.path.join(path, model_name + "_last.pth"))


def parse_utils(parser):
    parser.add_argument('--src',
                        help='source language',
                        type=str,
                        default='en')
    parser.add_argument('--trg',
                        help='target language',
                        type=str,
                        default='es')
    parser.add_argument('--token_min_freq',
                        help='minimum frequency for a token to be included in vocabulary, if excluded it is <unk>',
                        type=int,
                        default=1)
    parser.add_argument('--checkpoint_dir',
                        help='directory where models will be saved and loaded from',
                        type=str,
                        default='checkpoints')
    parser.add_argument('--load_model_name',
                        help='name of the model to load inside checkpoint_dir',
                        type=str,
                        default=None)
    parser.add_argument('--test',
                        help='perform test on testing and long testing sets, model will be loaded from checkpoint_dir',
                        default=False,
                        action="store_true")
    parser.add_argument('--batch_size',
                        help='mini batch size',
                        type=int,
                        default=64)
    parser.add_argument('--epochs',
                        help='number of epochs',
                        type=int,
                        default=30)
    parser.add_argument('--clip',
                        help='gradient clipping',
                        type=float,
                        default=1.0)
    parser.add_argument('--lr',
                        help='learning_rate',
                        type=float,
                        default=1e-3)
    parser.add_argument('--weight_decay',
                        help='weight_decay',
                        type=float,
                        default=1e-5)
