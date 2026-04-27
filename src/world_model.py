"""WorldModel: encoder + RSSM + decoder + Dreamer-style training loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import kl_divergence

from src.encoder import Decoder, Encoder
from src.rssm import RSSM


class WorldModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        h_dim: int = 200,
        z_dim: int = 30,
        action_dim: int = 2,
        hidden_dim: int = 200,
        min_std: float = 0.1,
        kl_weight: float = 1.0,
        free_nats: float = 3.0,
    ):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        self.rssm = RSSM(
            h_dim=h_dim,
            z_dim=z_dim,
            action_dim=action_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            min_std=min_std,
        )
        self.decoder = Decoder(feature_dim=h_dim + z_dim)
        self.kl_weight = kl_weight
        self.free_nats = free_nats

    def observe(self, obs_seq: torch.Tensor, action_seq: torch.Tensor):
        """Run the posterior forward over (B, T, ...) sequences.

        Returns h_seq, z_seq, post_dists, prior_dists, where the dist objects
        are lists of length T of torch.distributions.Normal (one per timestep).
        """
        B, T = obs_seq.shape[:2]
        device = obs_seq.device
        # Encode all frames in one pass for speed.
        flat_obs = obs_seq.reshape(B * T, *obs_seq.shape[2:])
        flat_e = self.encoder(flat_obs)
        e_seq = flat_e.view(B, T, -1)

        h, z = self.rssm.init_state(batch_size=B, device=device)
        h_list, z_list, post_list, prior_list = [], [], [], []
        for t in range(T):
            h, z, post, prior = self.rssm.forward_posterior(h, z, action_seq[:, t], e_seq[:, t])
            h_list.append(h)
            z_list.append(z)
            post_list.append(post)
            prior_list.append(prior)
        h_seq = torch.stack(h_list, dim=1)
        z_seq = torch.stack(z_list, dim=1)
        return h_seq, z_seq, post_list, prior_list

    def imagine(self, init_state, action_seq: torch.Tensor):
        """Run the prior forward, no observations."""
        h, z = init_state
        T = action_seq.shape[1]
        h_list, z_list = [], []
        for t in range(T):
            h, z, _ = self.rssm.forward_prior(h, z, action_seq[:, t])
            h_list.append(h)
            z_list.append(z)
        return torch.stack(h_list, dim=1), torch.stack(z_list, dim=1)

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        feature = torch.cat([h, z], dim=-1)
        flat = feature.reshape(-1, feature.shape[-1])
        flat_img = self.decoder(flat)
        return flat_img.view(*feature.shape[:-1], 3, 64, 64)

    def loss(self, obs_seq: torch.Tensor, action_seq: torch.Tensor) -> dict:
        h_seq, z_seq, post_list, prior_list = self.observe(obs_seq, action_seq)
        recon = self.decode(h_seq, z_seq)
        recon_loss = ((recon - obs_seq) ** 2).mean()

        kl_per_step = torch.stack(
            [kl_divergence(q, p).sum(dim=-1) for q, p in zip(post_list, prior_list)],
            dim=1,
        )  # (B, T)
        raw_kl = kl_per_step.mean()
        kl_loss = torch.clamp(raw_kl, min=self.free_nats)

        post_entropy = torch.stack([q.entropy().sum(dim=-1).mean() for q in post_list]).mean()
        prior_entropy = torch.stack([p.entropy().sum(dim=-1).mean() for p in prior_list]).mean()

        total = recon_loss + self.kl_weight * kl_loss
        return {
            "loss": total,
            "recon_loss": recon_loss.detach(),
            "kl_loss": raw_kl.detach(),
            "posterior_entropy": post_entropy.detach(),
            "prior_entropy": prior_entropy.detach(),
        }
