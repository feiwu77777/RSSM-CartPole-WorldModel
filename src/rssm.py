"""Recurrent State Space Model: deterministic GRU + stochastic Gaussian latent.

Per Dreamer (Hafner et al. 2019). The latent state at each step is the
concatenation (h_t, z_t):
  - h_t is a GRU hidden state advanced from (h_{t-1}, z_{t-1}, a_{t-1})
  - z_t is a Gaussian sample, drawn either from the prior p(z_t | h_t)
    (used when imagining without observations) or the posterior
    q(z_t | h_t, e_t) where e_t = Encoder(o_t) (used during training).
"""

import torch
from torch import nn
from torch.distributions import Normal


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ELU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class RSSM(nn.Module):
    def __init__(
        self,
        h_dim: int = 200,
        z_dim: int = 30,
        action_dim: int = 2,
        embed_dim: int = 1024,
        hidden_dim: int = 200,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.min_std = min_std

        self.gru_input_proj = _build_mlp(z_dim + action_dim, hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=h_dim)

        self.prior_net = _build_mlp(h_dim, hidden_dim, 2 * z_dim)
        self.post_net = _build_mlp(h_dim + embed_dim, hidden_dim, 2 * z_dim)

    def init_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def _gaussian(self, params: torch.Tensor) -> Normal:
        mu, raw = params.chunk(2, dim=-1)
        std = torch.nn.functional.softplus(raw) + self.min_std
        return Normal(mu, std)

    def _step_h(self, h_prev: torch.Tensor, z_prev: torch.Tensor, a_prev: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_prev, a_prev], dim=-1)
        x = self.gru_input_proj(x)
        return self.gru(x, h_prev)

    def forward_prior(self, h_prev: torch.Tensor, z_prev: torch.Tensor, a_prev: torch.Tensor):
        h = self._step_h(h_prev, z_prev, a_prev)
        prior = self._gaussian(self.prior_net(h))
        z = prior.rsample()
        return h, z, prior

    def forward_posterior(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        a_prev: torch.Tensor,
        e: torch.Tensor,
    ):
        h = self._step_h(h_prev, z_prev, a_prev)
        prior = self._gaussian(self.prior_net(h))
        post = self._gaussian(self.post_net(torch.cat([h, e], dim=-1)))
        z = post.rsample()
        return h, z, post, prior
