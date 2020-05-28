import math
import warnings

import torch
from torch.nn.modules import Module
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import (
    MultivariateNormal,
    _batch_mahalanobis,
)
from torch.distributions.normal import Normal


def _standard_normal_quantile(u):
    # Ref: https://en.wikipedia.org/wiki/Normal_distribution
    return math.sqrt(2) * torch.erfinv(2 * u - 1)


def _standard_normal_cdf(x):
    # Ref: https://en.wikipedia.org/wiki/Normal_distribution
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class GaussianCopula(Distribution):
    r"""
    A Gaussian copula.
    Args:
        covariance_matrix (torch.Tensor): positive-definite covariance matrix
    """
    arg_constraints = {"covariance_matrix": constraints.positive_definite}
    support = constraints.interval(0.0, 1.0)
    has_rsample = True

    def __init__(self, covariance_matrix=None, validate_args=None):
        # convert the covariance matrix to the correlation matrix
        # self.covariance_matrix = covariance_matrix.clone()
        # batch_diag = torch.diagonal(self.covariance_matrix, dim1=-1, dim2=-2).pow(-0.5)
        # self.covariance_matrix *= batch_diag.unsqueeze(-1)
        # self.covariance_matrix *= batch_diag.unsqueeze(-2)
        diag = torch.diag(covariance_matrix).pow(-0.5)
        self.covariance_matrix = (
            torch.diag(diag)).matmul(covariance_matrix).matmul(
            torch.diag(diag))

        batch_shape, event_shape = (
            covariance_matrix.shape[:-2],
            covariance_matrix.shape[-1:],
        )

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        self.multivariate_normal = MultivariateNormal(
            loc=torch.zeros(event_shape),
            covariance_matrix=self.covariance_matrix,
            validate_args=validate_args,
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value_x = _standard_normal_quantile(value)
        half_log_det = (
            self.multivariate_normal._unbroadcasted_scale_tril.diagonal(
                dim1=-2, dim2=-1
            )
            .log()
            .sum(-1)
        )
        M = _batch_mahalanobis(
            self.multivariate_normal._unbroadcasted_scale_tril, value_x
        )
        M -= value_x.pow(2).sum(-1)
        return -0.5 * M - half_log_det


def nll_copula(pred, label, cov):
    n_copula = GaussianCopula(cov)
    n_pred = Normal(loc=pred, scale=torch.diag(cov).pow(0.5))
    u = torch.clamp(n_pred.cdf(label), 0.01, 0.99)
    return -n_copula.log_prob(u)


class CopulaLoss(Module):
    __constants__ = ['reduction']

    def __init__(self, reduction='mean'):
        super(CopulaLoss, self).__init__()
        self.requires_cov = "cov"
        self.reduction = reduction

    def forward(self, pred, target, sigma):
        if not (target.size() == pred.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(target.size(), pred.size()),
                          stacklevel=2)
        pred, target = torch.broadcast_tensors(pred, target)
        sigma = sigma.float()
        nll_c = nll_copula(pred, target, sigma)
        normal = Normal(loc=pred, scale=torch.diag(sigma).pow(0.5))
        nll_q = -normal.log_prob(target)
        return nll_c + nll_q.sum()
