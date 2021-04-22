import torch
import torch.nn as nn


class RLSTCriterion(nn.Module):

    def __init__(self, rho, pad_index):
        super().__init__()
        self.RHO = rho
        self.rho_to_n = 1  # n is minibatch index
        self.mistranslation_loss_weight = 0
        self.policy_loss_weight = 0
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.policy_criterion = nn.MSELoss(reduction="sum")

    def forward(self, word_outputs, trg, Q_used, Q_target, policy_divisor):
        mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)
        if self.training:
            policy_loss = self.policy_criterion(Q_used, Q_target)/torch.count_nonzero(Q_target)
            self.rho_to_n *= self.RHO
            w_k = (self.RHO - self.rho_to_n) / (1 - self.rho_to_n)
            self.mistranslation_loss_weight = w_k * self.mistranslation_loss_weight + (1 - w_k) * float(mistranslation_loss)
            self.policy_loss_weight = w_k * self.policy_loss_weight + (1 - w_k) * float(policy_loss)
            loss = policy_loss / (self.policy_loss_weight * policy_divisor) + mistranslation_loss / self.mistranslation_loss_weight
            return loss, mistranslation_loss, policy_loss
        else:
            return None, mistranslation_loss, None