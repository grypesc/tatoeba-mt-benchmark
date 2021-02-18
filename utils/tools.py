from torch import zeros


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def int_to_one_hot(index: int, length, device):
    t = zeros(length, device=device)
    t[index] = 1
    return t
