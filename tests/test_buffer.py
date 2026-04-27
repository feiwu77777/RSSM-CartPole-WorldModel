import numpy as np

from src.buffer import SequenceReplayBuffer


def _fake_episode(length: int) -> dict:
    return {
        "obs": np.random.rand(length + 1, 3, 64, 64).astype(np.float32),
        "actions": np.random.randint(0, 2, size=(length,)).astype(np.int64),
        "dones": np.zeros(length, dtype=bool),
    }


def test_sample_shapes():
    buf = SequenceReplayBuffer(capacity=10, action_dim=2)
    for _ in range(3):
        buf.add(_fake_episode(60))
    batch = buf.sample(batch_size=4, seq_len=20, rng=np.random.default_rng(0))
    assert batch["obs"].shape == (4, 20, 3, 64, 64)
    assert batch["actions"].shape == (4, 20, 2)
    assert batch["dones"].shape == (4, 20)


def test_capacity_evicts_oldest():
    buf = SequenceReplayBuffer(capacity=2, action_dim=2)
    e1 = _fake_episode(10)
    e2 = _fake_episode(10)
    e3 = _fake_episode(10)
    buf.add(e1)
    buf.add(e2)
    buf.add(e3)
    assert len(buf) == 2


def test_sample_skips_too_short_episodes():
    buf = SequenceReplayBuffer(capacity=10, action_dim=2)
    buf.add(_fake_episode(5))   # too short for seq_len=20
    buf.add(_fake_episode(60))  # long enough
    batch = buf.sample(batch_size=2, seq_len=20, rng=np.random.default_rng(0))
    assert batch["obs"].shape == (2, 20, 3, 64, 64)


def test_chunks_are_contiguous():
    buf = SequenceReplayBuffer(capacity=10, action_dim=2)
    ep = _fake_episode(50)
    # Mark obs uniquely so we can verify contiguity
    for t in range(ep["obs"].shape[0]):
        ep["obs"][t, 0, 0, 0] = float(t)
    buf.add(ep)
    batch = buf.sample(batch_size=1, seq_len=10, rng=np.random.default_rng(1))
    times = batch["obs"][0, :, 0, 0, 0]
    diffs = np.diff(times)
    assert np.allclose(diffs, 1.0)
