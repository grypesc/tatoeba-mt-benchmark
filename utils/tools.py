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


def bleu(output, target, vocab):
    _, output_indices = torch.max(output, 2)
    output_indices = output_indices.transpose(1, 0)
    target = target.transpose(1, 0)
    target_str = [[vocab.itos[x] for x in y] for y in target]
    target_str = [[x[:x.index("<eos>")]] for x in target_str]
    output_str = [[vocab.itos[x] for x in y] for y in output_indices]
    output_str = [x[:len(target_str[i][0])] for i, x in enumerate(output_str)]
    return bleu_score(output_str, target_str)


def actions_ratio(actions):
    s = sum(actions)
    a = [actions[0]/s, actions[1]/s, actions[2]/s]
    return [round(action, 2) for action in a]
