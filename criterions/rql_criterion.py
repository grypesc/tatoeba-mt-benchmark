import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


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
            return loss, _mistranslation_loss, _policy_loss
        else:
            return None, _mistranslation_loss, None


class RQLCriterionExp(nn.Module):

    def __init__(self, ro, pad_index, mistranslation_loss_multiplier):
        super().__init__()
        self.RO = ro
        self.MISTRANSLATION_LOSS_MULTIPLIER = mistranslation_loss_multiplier
        self.ro_to_k = 1
        self._mistranslation_loss_weight = 0
        self._policy_loss_weight = 0
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.policy_criterion = nn.MSELoss(reduction="sum")
        self.writer = SummaryWriter(log_dir="logs/rql/" + "MISTRANSLATION_MULTIPLIER=" + str(mistranslation_loss_multiplier))
        self.total_iteration = 0
        self.n = 0

    def forward(self, word_outputs, trg, Q_used, Q_target):
        _mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)

        if self.training:
            self.n += 1
            _policy_loss = self.policy_criterion(Q_used, Q_target)/torch.count_nonzero(Q_target)
            self.ro_to_k *= self.RO
            w_k = (self.RO - self.ro_to_k) / (1 - self.ro_to_k)
            self._mistranslation_loss_weight = w_k * self._mistranslation_loss_weight + (1 - w_k) * float(_mistranslation_loss)
            self._policy_loss_weight = (1 - 1/self.n) * self._policy_loss_weight + float(_policy_loss) / self.n
            loss = _policy_loss / self._policy_loss_weight + self.MISTRANSLATION_LOSS_MULTIPLIER * _mistranslation_loss / self._mistranslation_loss_weight
            self.total_iteration += 1
            self.writer.add_scalar('loss_weights/policy', self._policy_loss_weight, self.total_iteration)
            self.writer.add_scalar('loss_weights/mistranslation', self._mistranslation_loss_weight, self.total_iteration)
            self.writer.add_scalar('loss/policy', _policy_loss, self.total_iteration)
            self.writer.add_scalar('loss/mistranslation', _mistranslation_loss, self.total_iteration)
            self.writer.add_scalar('total/policy', _policy_loss / self._policy_loss_weight, self.total_iteration)
            self.writer.add_scalar('total/mistranslation', self.MISTRANSLATION_LOSS_MULTIPLIER * _mistranslation_loss / self._mistranslation_loss_weight, self.total_iteration)
            return loss, _mistranslation_loss, _policy_loss
        else:
            return None, _mistranslation_loss, None