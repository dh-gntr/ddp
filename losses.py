import numpy as np
import math
import torch
import torch.nn as nn
from torch import digamma
import torch.nn.functional as F

class EDLLoss(nn.Module):
    def __init__(self,initial):
        super(EDLLoss, self).__init__()
        self.initial = initial

    # Expected cross-entropy (Eq.4 from paper)
    def expected_cross_entropy(self,alpha, target_onehot):
        # alpha: [B, K], target_onehot: [B, K]
        S = torch.sum(alpha, dim=1, keepdim=True)             # [B,1]
        digamma_S = torch.digamma(S)
        digamma_alpha = torch.digamma(alpha)
        # per-sample loss sum_k y_k (psi(S) - psi(alpha_k))
        loss = torch.sum(target_onehot * (digamma_S - digamma_alpha), dim=1)
        return loss.mean()

    # KL Divergence between Dir(alpha_tilde) and Dir(ones)
    def dirichlet_kl(self,alpha_tilde):
        # alpha_tilde: [B, K]
        k = alpha_tilde.size(1)
        alpha0 = torch.sum(alpha_tilde, dim=1, keepdim=True) # [B,1]
        term1 = torch.lgamma(alpha0) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
        term2 = - (k * math.lgamma(1.0) - k * math.lgamma(1.0))  # zeros, but keep for clarity
        # second piece
        psi_alpha = torch.digamma(alpha_tilde)
        psi_alpha0 = torch.digamma(alpha0)
        term3 = ((alpha_tilde - 1.0) * (psi_alpha - psi_alpha0)).sum(dim=1, keepdim=True)
        kl = (term1 + term2 + term3).squeeze(1)
        return kl.mean()

    # Full loss with annealed KL
    def forward(self,logits, target, epoch, kl_coefficient_max=1.0):
        # target: integer labels [B]
        
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        B, K = logits.shape
        # one-hot
        target_onehot = F.one_hot(target, num_classes=K).float()
        # Expected CE
        err_loss = self.expected_cross_entropy(alpha, target_onehot)
        # Build alpha_tilde = y + (1-y) * alpha  (remove non-misleading evidence)
        alpha_tilde = target_onehot + (1.0 - target_onehot) * alpha
        # KL
        kl = self.dirichlet_kl(alpha_tilde)
        # anneal
        lambda_t = min(kl_coefficient_max, epoch / 10.0)   # paper uses min(1.0, t/10)
        loss = err_loss + lambda_t * kl
        return loss #, err_loss.detach(), kl.detach()

# Reference: https://github.com/tjoo512/belief-matching-framework
class BeliefMatchingLoss(nn.Module):
    def __init__(self, coeff, prior=1.):
        super(BeliefMatchingLoss, self).__init__()
        self.prior = prior
        self.coeff = coeff

    def forward(self, logits, ys,logits1,logits2):
        evidence = F.softplus(logits)
        alphas = evidence + 1.0
        #alphas = torch.exp(logits)
        betas = self.prior * torch.ones_like(logits)
        # alpha_hats = torch.ones_like(logits) * self.prior + torch.nn.functional.one_hot(ys, num_classes=10)
        # return kl_div_dirichlets(alphas, alpha_hats)

        # compute log-likelihood loss: psi(alpha_target) - psi(alpha_zero)
        #print(ys.shape)
        a_zero = torch.sum(alphas, -1)
        #if ys[0] != 10:
        a_ans = torch.gather(alphas, -1, ys.unsqueeze(-1)).squeeze(-1)
        #a_zero = torch.sum(alphas, -1)
        ll_loss = digamma(a_ans) - digamma(a_zero)

        # compute kl loss: loss1 + loss2
        #       loss1 = log_gamma(alpha_zero) - \sum_k log_gamma(alpha_zero)
        #       loss2 = sum_k (alpha_k - beta_k) (digamma(alpha_k) - digamma(alpha_zero) )
        loss1 = torch.lgamma(a_zero) - torch.sum(torch.lgamma(alphas), -1)
        loss2 = torch.sum(
            (alphas - betas) * (digamma(alphas) - digamma(a_zero.unsqueeze(-1))),
            -1)
        kl_loss = loss1 + loss2
        
        log_p = F.log_softmax(logits1, dim=1)
        p = F.softmax(logits1, dim=1)

        log_q = F.log_softmax(logits2, dim=1)
        q = F.softmax(logits2, dim=1)

        kl_pq = F.kl_div(p, q, reduction='batchmean')
        kl_qp = F.kl_div(q, p, reduction='batchmean')

        # if ys[0] == 10 :
        #     loss = (self.coeff * kl_loss -0.5*(kl_pq+kl_qp)).mean()
        # else:
        loss =((self.coeff * kl_loss - ll_loss + 0.5*(kl_pq+kl_qp) )).mean()
        return loss #((self.coeff * kl_loss - ll_loss + 0.01*interheadloss )).mean()


def betaln(alphas, dim=-1):
    return torch.sum(torch.lgamma(alphas), dim=dim) - torch.lgamma(torch.sum(alphas, dim=dim))


def kl_div_dirichlets(alphas, betas, dim=-1):
    alpha0 = alphas.sum(dim)
    beta0 = betas.sum(dim)
    t1 = alpha0.lgamma() - beta0.lgamma()
    t2 = (alphas.lgamma() - betas.lgamma()).sum(dim)
    t3 = alphas - betas
    t4 = alphas.digamma() - alpha0.digamma().unsqueeze(dim)
    return t1 - t2 + (t3 * t4).sum(dim)
