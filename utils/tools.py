import torch

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


def save_model(model, path, is_best, epoch):
    if is_best:
        torch.save(model.state_dict(), path + "_best_{}.pth".format(epoch))
    torch.save(model.state_dict(), path + "_last.pth")
