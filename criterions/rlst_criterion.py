import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class RLSTCriterion(nn.Module):

    def __init__(self, ro, pad_index):
        super().__init__()
        self.RO = ro
        self.ro_to_k = 1
        self.mistranslation_loss_weight = 0
        self.policy_loss_weight = 0
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.policy_criterion = nn.MSELoss(reduction="sum")

    def forward(self, word_outputs, trg, Q_used, Q_target, mistranslation_loss_multiplier):
        mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)
        if self.training:
            policy_loss = self.policy_criterion(Q_used, Q_target)/torch.count_nonzero(Q_target)
            self.ro_to_k *= self.RO
            w_k = (self.RO - self.ro_to_k) / (1 - self.ro_to_k)
            self.mistranslation_loss_weight = w_k * self.mistranslation_loss_weight + (1 - w_k) * float(mistranslation_loss)
            self.policy_loss_weight = w_k * self.policy_loss_weight + (1 - w_k) * float(policy_loss)
            loss = policy_loss / self.policy_loss_weight + mistranslation_loss_multiplier * mistranslation_loss / self.mistranslation_loss_weight
            return loss, mistranslation_loss, policy_loss
        else:
            return None, mistranslation_loss, None


class RLSTCriterionExp(nn.Module):

    def __init__(self, ro, pad_index, mistranslation_loss_multiplier):
        super().__init__()
        self.RO = ro
        self.MISTRANSLATION_LOSS_MULTIPLIER = mistranslation_loss_multiplier
        self.ro_to_k = 1
        self.mistranslation_loss_weight = 0
        self.policy_loss_weight = 0
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.policy_criterion = nn.MSELoss(reduction="sum")
        self.writer = SummaryWriter(log_dir="logs/rql/" + "MISTRANSLATION_MULTIPLIER=" + str(mistranslation_loss_multiplier))
        self.total_iteration = 0
        self.n = 0

    def forward(self, word_outputs, trg, Q_used, Q_target):
        mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)

        if self.training:
            self.n += 1
            policy_loss = self.policy_criterion(Q_used, Q_target)/torch.count_nonzero(Q_target)
            self.ro_to_k *= self.RO
            w_k = (self.RO - self.ro_to_k) / (1 - self.ro_to_k)
            self.mistranslation_loss_weight = w_k * self.mistranslation_loss_weight + (1 - w_k) * float(mistranslation_loss)
            self.policy_loss_weight = (1 - 1/self.n) * self.policy_loss_weight + float(policy_loss) / self.n
            loss = policy_loss / self.policy_loss_weight + self.MISTRANSLATION_LOSS_MULTIPLIER * mistranslation_loss / self.mistranslation_loss_weight
            self.total_iteration += 1
            self.writer.add_scalar('loss_weights/policy', self.policy_loss_weight, self.total_iteration)
            self.writer.add_scalar('loss_weights/mistranslation', self.mistranslation_loss_weight, self.total_iteration)
            self.writer.add_scalar('loss/policy', policy_loss, self.total_iteration)
            self.writer.add_scalar('loss/mistranslation', mistranslation_loss, self.total_iteration)
            self.writer.add_scalar('total/policy', policy_loss / self.policy_loss_weight, self.total_iteration)
            self.writer.add_scalar('total/mistranslation', self.MISTRANSLATION_LOSS_MULTIPLIER * mistranslation_loss / self.mistranslation_loss_weight, self.total_iteration)
            return loss, mistranslation_loss, policy_loss
        else:
            return None, mistranslation_loss, None


class RLSTCriterionV3(nn.Module):

    def __init__(self, ro, pad_index, mistranslation_loss_multiplier):
        super().__init__()
        self.RO = ro
        self.MISTRANSLATION_LOSS_MULTIPLIER = mistranslation_loss_multiplier
        self.ro_to_k = 1
        self.mistranslation_loss_weight = 0
        self.policy_loss_weight = 0
        self.policy_loss_weight_dashx2 = 0
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.policy_criterion = nn.MSELoss(reduction="sum")
        self.writer = SummaryWriter(log_dir="logs/rql/" + "V3MISTRANSLATION_MULTIPLIER=" + str(mistranslation_loss_multiplier))
        self.total_iteration = 0
        self.n = 0

    def forward(self, word_outputs, trg, Q_used, Q_target, _):
        mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)

        if self.training:
            self.n += 1
            policy_loss = self.policy_criterion(Q_used, Q_target)/torch.count_nonzero(Q_target)
            self.ro_to_k *= self.RO
            w_k = (self.RO - self.ro_to_k) / (1 - self.ro_to_k)
            self.mistranslation_loss_weight = w_k * self.mistranslation_loss_weight + (1 - w_k) * float(mistranslation_loss)
            self.policy_loss_weight = w_k * self.policy_loss_weight + (1 - w_k) * float(policy_loss)
            self.policy_loss_weight_dashx2 = (1 - 1/self.n) * self.policy_loss_weight_dashx2 + float(policy_loss) / self.n
            loss = policy_loss * self.policy_loss_weight / self.policy_loss_weight_dashx2 ** 2 + self.MISTRANSLATION_LOSS_MULTIPLIER * mistranslation_loss / self.mistranslation_loss_weight
            self.total_iteration += 1
            self.writer.add_scalar('loss_weights/policy', self.policy_loss_weight_dashx2, self.total_iteration)
            self.writer.add_scalar('loss_weights/mistranslation', self.mistranslation_loss_weight, self.total_iteration)
            self.writer.add_scalar('loss/policy', policy_loss, self.total_iteration)
            self.writer.add_scalar('loss/mistranslation', mistranslation_loss, self.total_iteration)
            self.writer.add_scalar('total/policy', policy_loss * self.policy_loss_weight / self.policy_loss_weight_dashx2 ** 2, self.total_iteration)
            self.writer.add_scalar('total/mistranslation', self.MISTRANSLATION_LOSS_MULTIPLIER * mistranslation_loss / self.mistranslation_loss_weight, self.total_iteration)
            return loss, mistranslation_loss, policy_loss
        else:
            return None, mistranslation_loss, None
