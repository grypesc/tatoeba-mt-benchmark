import torch
import torch.nn as nn


class RQLCriterion(nn.Module):

    def __init__(self, ro, pad_index, mistranslation_loss_multiplier):
        super().__init__()
        self.RO = ro
        self.MISTRANSLATION_LOSS_MULTIPLIER = mistranslation_loss_multiplier
        self.ro_to_k = 1
        self._mistranslation_loss_weight = 0
        self._policy_loss_weight = 0
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.policy_criterion = nn.MSELoss(reduction="sum")

    def forward(self, word_outputs, trg, Q_used, Q_target):
        _mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)

        if self.training:
            _policy_loss = self.policy_criterion(Q_used, Q_target)/torch.count_nonzero(Q_target)
            self.ro_to_k *= self.RO
            w_k = (self.RO - self.ro_to_k) / (1 - self.ro_to_k)
            self._mistranslation_loss_weight = w_k * self._mistranslation_loss_weight + (1 - w_k) * float(_mistranslation_loss)
            self._policy_loss_weight = w_k * self._policy_loss_weight + (1 - w_k) * float(_policy_loss)
            loss = _policy_loss / self._policy_loss_weight + self.MISTRANSLATION_LOSS_MULTIPLIER * _mistranslation_loss / self._mistranslation_loss_weight
        else:
            loss = -1

        return loss, _mistranslation_loss


class RQLNoEosCriterion(nn.Module):
    """
    This criterion increases cross entropy loss on sequence positions where instead of target eos, model outputed
    something  else.
    """
    def __init__(self, ro, pad_index, eos_index, mistranslation_loss_multiplier, no_eos_multiplier):
        super().__init__()
        self.ro = ro
        self.ro_to_k = 1
        self._mistranslation_loss_weight = 0
        self._policy_loss_weight = 0
        self.mistranslation_loss = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='none')
        self.policy_criterion = nn.MSELoss()
        self.mistranslation_loss_multiplier = mistranslation_loss_multiplier
        self.pad_index = pad_index
        self.eos_index = eos_index
        self.no_eos_multiplier = no_eos_multiplier

    def forward(self, word_outputs, trg, Q_used, Q_target):
        """
        The shape of word_outputs and trg is different compared to RQLCriterion
        """

        no_eos_outputs = (word_outputs.max(2)[1] != trg) * (trg == self.eos_index)
        word_outputs = word_outputs.view(-1, word_outputs.shape[-1])
        trg = trg.view(-1)
        _mistranslation_loss = self.mistranslation_loss(word_outputs, trg)
        _mistranslation_loss[no_eos_outputs.view(-1)] *= self.no_eos_multiplier
        _mistranslation_loss = torch.mean(_mistranslation_loss[trg != self.pad_index])

        if self.training:
            _policy_loss = self.policy_criterion(Q_used, Q_target)
            self.ro_to_k *= self.ro
            w_k = (self.ro - self.ro_to_k) / (1 - self.ro_to_k)
            self._mistranslation_loss_weight = w_k * self._mistranslation_loss_weight + (1 - w_k) * float(_mistranslation_loss)
            self._policy_loss_weight = w_k * self._policy_loss_weight + (1 - w_k) * float(_policy_loss)
            loss = _policy_loss / self._policy_loss_weight + self.mistranslation_loss_multiplier * _mistranslation_loss / self._mistranslation_loss_weight
        else:
            loss = -1

        return loss, _mistranslation_loss