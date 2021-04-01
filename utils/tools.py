import torch

from torch import zeros
from torchtext.data.metrics import bleu_score


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def int_to_one_hot(index: int, length, device):
    t = zeros(length, device=device)
    t[index] = 1
    return t


def bleu(output, target, vocab, device):
    _, output_words = torch.max(output, 2)
    batch_size = output_words.size()[1]
    eos = torch.tensor(vocab.stoi['<eos>']).to(device)
    eos_vector = torch.full((1, batch_size), eos, device=device)

    output_words = torch.cat((output_words, eos_vector), dim=0)
    output_is_eos = torch.eq(output_words, eos)
    output_eos = output_is_eos.max(0)[1]
    output_words = output_words.transpose(1, 0)

    trg_is_eos = torch.eq(target, eos)
    trg_eos = trg_is_eos.max(0)[1]

    target = target.transpose(1, 0)
    target = [x[:trg_eos[i]] for i, x in enumerate(target)]
    target_str = [[[vocab.itos[x] for x in y]] for y in target]
    output = [x[:output_eos[i]] for i, x in enumerate(output_words)]
    output_str = [[vocab.itos[x] for x in y] for y in output]
    return bleu_score(output_str, target_str)


def actions_ratio(actions):
    s = sum(actions)
    a = [actions[0]/s, actions[1]/s, actions[2]/s]
    return [round(action, 2) for action in a]
