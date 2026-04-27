import torch

from src.world_model import WorldModel


def _toy_world_model() -> WorldModel:
    return WorldModel(
        embed_dim=1024,
        h_dim=200,
        z_dim=30,
        action_dim=2,
        hidden_dim=200,
        min_std=0.1,
        kl_weight=1.0,
        free_nats=3.0,
    )


def test_loss_dict_has_expected_keys():
    wm = _toy_world_model()
    obs = torch.zeros(2, 5, 3, 64, 64)
    actions = torch.zeros(2, 5, 2)
    losses = wm.loss(obs, actions)
    assert set(losses.keys()) == {
        "loss",
        "recon_loss",
        "kl_loss",
        "posterior_entropy",
        "prior_entropy",
    }


def test_one_optim_step_decreases_loss():
    torch.manual_seed(0)
    wm = _toy_world_model()
    optim = torch.optim.Adam(wm.parameters(), lr=1e-3)
    obs = torch.rand(2, 5, 3, 64, 64)
    actions = torch.zeros(2, 5, 2)
    actions[:, :, 0] = 1.0

    initial = wm.loss(obs, actions)["loss"].item()
    for _ in range(5):
        optim.zero_grad()
        out = wm.loss(obs, actions)
        out["loss"].backward()
        optim.step()
    final = wm.loss(obs, actions)["loss"].item()
    assert final < initial, f"loss did not decrease: {initial:.4f} -> {final:.4f}"


def test_imagine_returns_correct_shapes():
    wm = _toy_world_model()
    h, z = wm.rssm.init_state(batch_size=2, device=torch.device("cpu"))
    actions = torch.zeros(2, 7, 2)
    h_seq, z_seq = wm.imagine((h, z), actions)
    assert h_seq.shape == (2, 7, 200)
    assert z_seq.shape == (2, 7, 30)
