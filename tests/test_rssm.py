import torch

from src.rssm import RSSM


def test_init_state_shapes():
    rssm = RSSM(h_dim=200, z_dim=30, action_dim=2, embed_dim=1024, hidden_dim=200)
    h, z = rssm.init_state(batch_size=4, device=torch.device("cpu"))
    assert h.shape == (4, 200)
    assert z.shape == (4, 30)


def test_forward_prior_shapes():
    rssm = RSSM(h_dim=200, z_dim=30, action_dim=2, embed_dim=1024, hidden_dim=200)
    h, z = rssm.init_state(batch_size=4, device=torch.device("cpu"))
    a = torch.zeros(4, 2)
    h_next, z_next, prior = rssm.forward_prior(h, z, a)
    assert h_next.shape == (4, 200)
    assert z_next.shape == (4, 30)
    assert prior.mean.shape == (4, 30)
    assert prior.stddev.shape == (4, 30)


def test_forward_posterior_shapes():
    rssm = RSSM(h_dim=200, z_dim=30, action_dim=2, embed_dim=1024, hidden_dim=200)
    h, z = rssm.init_state(batch_size=4, device=torch.device("cpu"))
    a = torch.zeros(4, 2)
    e = torch.zeros(4, 1024)
    h_next, z_next, post, prior = rssm.forward_posterior(h, z, a, e)
    assert h_next.shape == (4, 200)
    assert z_next.shape == (4, 30)
    assert post.mean.shape == (4, 30)
    assert prior.mean.shape == (4, 30)


def test_rollout_no_nans():
    torch.manual_seed(0)
    rssm = RSSM(h_dim=200, z_dim=30, action_dim=2, embed_dim=1024, hidden_dim=200)
    h, z = rssm.init_state(batch_size=2, device=torch.device("cpu"))
    a = torch.zeros(2, 2)
    for _ in range(50):
        h, z, _ = rssm.forward_prior(h, z, a)
    assert torch.isfinite(h).all()
    assert torch.isfinite(z).all()


def test_kl_zero_for_identical_distributions():
    from torch.distributions import Normal, kl_divergence

    mu = torch.zeros(4, 30)
    std = torch.ones(4, 30)
    p = Normal(mu, std)
    q = Normal(mu.clone(), std.clone())
    kl = kl_divergence(q, p).sum(-1)
    assert torch.allclose(kl, torch.zeros(4), atol=1e-6)
