import torch
import torch.nn.functional as F


class LabelSmoothedCrossEntropy(torch.nn.Module):
    def __init__(self, label_smoothing=0.0, ignore_index=None):
        super(LabelSmoothedCrossEntropy, self).__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def label_smoothed_nll_loss(self, lprobs, target, label_smoothing, reduce=True):
        """Taken from fairseq and added pad_indices_count as we have to calculate mean loss and therefore ignore pad indices"""
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        pad_count = 0
        if self.ignore_index is not None:
            pad_mask = target.eq(self.ignore_index)
            pad_count = pad_mask.sum()
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)

        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        if reduce:
            non_pad_count = lprobs.size()[0] - pad_count
            nll_loss = nll_loss.sum() / non_pad_count
            smooth_loss = smooth_loss.sum() / non_pad_count
        lab_sm_i = label_smoothing / (lprobs.size(-1) - 1)
        loss = (1.0 - label_smoothing - lab_sm_i) * nll_loss + lab_sm_i * smooth_loss
        return loss, nll_loss

    def forward(self, net_output, target, reduce):
        lprobs = torch.nn.functional.log_softmax(net_output, dim=-1)
        loss, nll_loss = self.label_smoothed_nll_loss(lprobs, target, self.label_smoothing, reduce=reduce)
        return loss


