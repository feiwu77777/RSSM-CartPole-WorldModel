"""Sequence replay buffer: stores whole episodes, samples contiguous chunks."""

from __future__ import annotations

from collections import deque

import numpy as np


def _one_hot(actions: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((actions.shape[0], n), dtype=np.float32)
    out[np.arange(actions.shape[0]), actions] = 1.0
    return out


class SequenceReplayBuffer:
    def __init__(self, capacity: int, action_dim: int):
        self.capacity = capacity
        self.action_dim = action_dim
        self.episodes: deque[dict] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, episode: dict) -> None:
        self.episodes.append(episode)

    def sample(self, batch_size: int, seq_len: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        eligible = [i for i, ep in enumerate(self.episodes) if ep["actions"].shape[0] >= seq_len]
        if len(eligible) == 0:
            raise ValueError(f"No episodes long enough for seq_len={seq_len}")
        chosen = rng.choice(eligible, size=batch_size, replace=True)

        obs_chunks = []
        action_chunks = []
        done_chunks = []
        for idx in chosen:
            ep = self.episodes[idx]
            T = ep["actions"].shape[0]
            start = int(rng.integers(0, T - seq_len + 1))
            end = start + seq_len
            obs_chunks.append(ep["obs"][start:end])
            action_chunks.append(_one_hot(ep["actions"][start:end], self.action_dim))
            done_chunks.append(ep["dones"][start:end])
        return {
            "obs": np.stack(obs_chunks, axis=0),
            "actions": np.stack(action_chunks, axis=0),
            "dones": np.stack(done_chunks, axis=0),
        }
