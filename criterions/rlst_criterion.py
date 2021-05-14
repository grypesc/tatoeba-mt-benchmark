import torch
import torch.nn as nn
import math


class RLSTCriterion(nn.Module):

    def __init__(self, rho, pad_index, N, eta_min, eta_max):
        super().__init__()
        self.RHO = rho
        self.rho_to_n = 1  # n is minibatch index
        self.n = -1
        self.N = N
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.mistranslation_loss_weight = 0
        self.policy_loss_weight = 0
        self.mistranslation_criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
        self.policy_criterion = nn.MSELoss(reduction="sum")
        self.policy_multiplier = 0

    def forward(self, word_outputs, trg, Q_used, Q_target):
        mistranslation_loss = self.mistranslation_criterion(word_outputs, trg)
        if self.training:
            self.n += 1
            self.policy_multiplier = self.eta_max - (self.eta_max - self.eta_min) * math.e ** ((-3) * self.n / self.N)
            policy_loss = self.policy_criterion(Q_used, Q_target)/torch.count_nonzero(Q_target)
            self.rho_to_n *= self.RHO
            w_k = (self.RHO - self.rho_to_n) / (1 - self.rho_to_n)
            self.mistranslation_loss_weight = w_k * self.mistranslation_loss_weight + (1 - w_k) * float(mistranslation_loss)
            self.policy_loss_weight = w_k * self.policy_loss_weight + (1 - w_k) * float(policy_loss)
            loss = policy_loss * self.policy_multiplier / self.policy_loss_weight + mistranslation_loss / self.mistranslation_loss_weight
            return loss, mistranslation_loss, policy_loss, self.policy_multiplier
        else:
            return None, mistranslation_loss, None, None
