# Tiny RSSM World Model on CartPole — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal Dreamer-style RSSM (deterministic GRU + stochastic Gaussian latent), train it on 64×64 pixel CartPole observations, and produce posterior reconstruction + imagined-rollout videos plus a latent-space PCA plot.

**Architecture:** A clean `src/` Python package: `encoder.py` (CNN + ConvTranspose decoder), `rssm.py` (GRU + prior + posterior MLPs), `world_model.py` (glue + reconstruction-and-KL loss), `env.py` (CartPolePixels + PDController), `buffer.py` (SequenceReplayBuffer), and `train.py`/`eval.py` CLI entry points reading `config.yaml`. Mirrors the layout of the sibling `Deep-Q-learning-Rat-Cheese` project.

**Tech Stack:** Python 3.10+, PyTorch ≥2.0, gymnasium[classic-control], OpenCV (`cv2`), scikit-video (`skvideo`), PyYAML, matplotlib, scikit-learn (PCA only), pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `.gitignore` | Create | Ignore `__pycache__/`, `model/`, `video/`, `figures/`, `.pytest_cache/` |
| `requirements.txt` | Create | Python dependencies |
| `config.yaml` | Create | All hyperparameters |
| `src/__init__.py` | Create | Package marker |
| `src/encoder.py` | Create | `Encoder` (CNN) + `Decoder` (ConvTranspose) |
| `src/rssm.py` | Create | `RSSM` — GRUCell + prior MLP + posterior MLP |
| `src/world_model.py` | Create | `WorldModel` — encoder + RSSM + decoder + loss |
| `src/env.py` | Create | `CartPolePixels` wrapper + `PDController` + `collect_episode` |
| `src/buffer.py` | Create | `SequenceReplayBuffer` (per-episode storage, chunk sampling) |
| `src/viz.py` | Create | mp4 + matplotlib helpers shared by train/eval |
| `src/train.py` | Create | Training entry point (`python -m src.train`) |
| `src/eval.py` | Create | Evaluation entry point (`python -m src.eval`) |
| `tests/__init__.py` | Create | Test package marker |
| `tests/test_encoder.py` | Create | Encoder/Decoder shape tests |
| `tests/test_rssm.py` | Create | Prior/posterior shapes, KL sanity, no-NaN rollout |
| `tests/test_env.py` | Create | Observation shape/dtype, PD balances ≥100 steps |
| `tests/test_buffer.py` | Create | Sample shape, contiguity within episode |
| `tests/test_world_model.py` | Create | One Adam step decreases loss, loss-dict keys |
| `docs/rssm_explained.md` | Create | Plain-English RSSM companion (analogue of `bellman_equation.md`) |
| `README.md` | Create | Short intro + numbered sections + result placeholders |

---

### Task 1: Project Scaffold

**Files:**
- Create: `.gitignore`, `requirements.txt`, `config.yaml`
- Create: `src/__init__.py`, `tests/__init__.py`
- Create directories: `model/`, `video/posterior/`, `video/rollouts/`, `figures/`

- [ ] **Step 1: Initialize git and create directories**

Run from project root `RSSM-CartPole-WorldModel/`:

```bash
git init
mkdir -p src tests model video/posterior video/rollouts figures
touch src/__init__.py tests/__init__.py
```

Expected: `.git/` directory created, source/test packages and runtime output directories exist.

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
.DS_Store
model/
video/
figures/
*.egg-info/
.venv/
venv/
```

- [ ] **Step 3: Create `requirements.txt`**

```
torch>=2.0
numpy
gymnasium[classic-control]
opencv-python
scikit-video
pyyaml
matplotlib
scikit-learn
pytest
```

- [ ] **Step 4: Create `config.yaml`**

```yaml
env:
  image_size: 64
  max_steps: 200
  expert_prob: 0.5
  pd_kp: 10.0
  pd_kd: 1.0

model:
  embed_dim: 1024
  h_dim: 200
  z_dim: 30
  hidden_dim: 200
  action_dim: 2
  min_std: 0.1

buffer:
  capacity: 200
  init_episodes: 50
  seq_len: 50

train:
  batch_size: 16
  train_steps: 20000
  lr: 6.0e-4
  adam_eps: 1.0e-4
  grad_clip: 100.0
  kl_weight: 1.0
  free_nats: 3.0
  collect_every: 5
  log_every: 100
  checkpoint_every: 1000
  video_every: 1000
  model_dir: model/
  video_dir: video/
  figure_dir: figures/

eval:
  checkpoint: model/world_model.pt
  num_episodes: 5
  context_len: 5
  imagine_horizon: 45
  video_dir: video/
  figure_dir: figures/
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt config.yaml src/__init__.py tests/__init__.py
git commit -m "chore: scaffold project structure and config"
```

---

### Task 2: Encoder & Decoder (TDD)

**Files:**
- Test: `tests/test_encoder.py`
- Create: `src/encoder.py`

- [ ] **Step 1: Write failing test for encoder shape**

Create `tests/test_encoder.py`:

```python
import torch

from src.encoder import Decoder, Encoder


def test_encoder_output_shape():
    enc = Encoder(embed_dim=1024)
    x = torch.zeros(4, 3, 64, 64)
    out = enc(x)
    assert out.shape == (4, 1024)


def test_decoder_output_shape():
    dec = Decoder(feature_dim=230)
    feat = torch.zeros(4, 230)
    out = dec(feat)
    assert out.shape == (4, 3, 64, 64)


def test_encoder_decoder_pipeline_runs():
    enc = Encoder(embed_dim=1024)
    dec = Decoder(feature_dim=230)
    x = torch.zeros(2, 3, 64, 64)
    e = enc(x)
    assert e.shape == (2, 1024)
    fake_state = torch.zeros(2, 230)
    y = dec(fake_state)
    assert y.shape == x.shape
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_encoder.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.encoder'`.

- [ ] **Step 3: Implement `src/encoder.py`**

```python
"""Pixel encoder and decoder for 64x64 RGB observations.

Mirrors the Dreamer small-image encoder: 4 stride-2 conv layers (no padding),
producing 256x2x2 = 1024 features after flatten. The decoder mirrors this
structure with ConvTranspose2d.
"""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        return h.flatten(start_dim=1)


class Decoder(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, 1024)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        h = self.fc(feature)
        h = h.view(-1, 1024, 1, 1)
        return self.deconv(h)
```

- [ ] **Step 4: Run test, verify it passes**

```bash
pytest tests/test_encoder.py -v
```

Expected: PASS for all three tests. If shapes don't line up, adjust `kernel_size` on the deconv layers (the asymmetric kernels above are tuned to invert the encoder's `(64→31→14→6→2)` flow back to 64).

- [ ] **Step 5: Commit**

```bash
git add src/encoder.py tests/test_encoder.py
git commit -m "feat: add CNN encoder and decoder for 64x64 observations"
```

---

### Task 3: RSSM (TDD)

**Files:**
- Test: `tests/test_rssm.py`
- Create: `src/rssm.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_rssm.py`:

```python
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_rssm.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/rssm.py`**

```python
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
```

- [ ] **Step 4: Run test, verify it passes**

```bash
pytest tests/test_rssm.py -v
```

Expected: PASS for all five tests.

- [ ] **Step 5: Commit**

```bash
git add src/rssm.py tests/test_rssm.py
git commit -m "feat: add RSSM with deterministic GRU and stochastic Gaussian latent"
```

---

### Task 4: Environment Wrapper + PD Controller (TDD)

**Files:**
- Test: `tests/test_env.py`
- Create: `src/env.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_env.py`:

```python
import numpy as np

from src.env import CartPolePixels, PDController, collect_episode


def test_observation_shape_and_dtype():
    env = CartPolePixels(image_size=64)
    obs = env.reset(seed=0)
    assert obs.shape == (3, 64, 64)
    assert obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0
    env.close()


def test_step_returns_correct_types():
    env = CartPolePixels(image_size=64)
    env.reset(seed=0)
    obs, reward, done = env.step(0)
    assert obs.shape == (3, 64, 64)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()


def test_pd_controller_balances_at_least_100_steps():
    env = CartPolePixels(image_size=64)
    env.reset(seed=0)
    ctrl = PDController(kp=10.0, kd=1.0)
    steps = 0
    done = False
    while not done and steps < 200:
        action = ctrl.act(env.unwrapped_state())
        _, _, done = env.step(action)
        steps += 1
    env.close()
    assert steps >= 100, f"PD controller balanced only {steps} steps"


def test_collect_episode_returns_expected_keys():
    env = CartPolePixels(image_size=64)
    ctrl = PDController(kp=10.0, kd=1.0)
    rng = np.random.default_rng(0)
    ep = collect_episode(env, ctrl, max_steps=50, expert_prob=1.0, rng=rng)
    env.close()
    assert set(ep.keys()) == {"obs", "actions", "dones"}
    T = len(ep["actions"])
    assert ep["obs"].shape == (T + 1, 3, 64, 64)
    assert ep["actions"].shape == (T,)
    assert ep["dones"].shape == (T,)
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_env.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/env.py`**

```python
"""CartPole pixel-observation wrapper, PD controller, and episode collector."""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np


class CartPolePixels:
    def __init__(self, image_size: int = 64):
        self.image_size = image_size
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")

    def _render(self) -> np.ndarray:
        frame = self.env.render()  # (H, W, 3) uint8
        frame = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        frame = frame.astype(np.float32) / 255.0
        return np.transpose(frame, (2, 0, 1))  # (3, H, W)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.env.reset(seed=seed)
        return self._render()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        _, reward, terminated, truncated, _ = self.env.step(int(action))
        done = bool(terminated or truncated)
        return self._render(), float(reward), done

    def unwrapped_state(self) -> np.ndarray:
        # gymnasium stores [pos, vel, angle, ang_vel] on the underlying env
        return np.asarray(self.env.unwrapped.state, dtype=np.float32)

    def close(self) -> None:
        self.env.close()


class PDController:
    def __init__(self, kp: float = 10.0, kd: float = 1.0):
        self.kp = kp
        self.kd = kd

    def act(self, state: np.ndarray) -> int:
        # state = [pos, vel, angle, ang_vel]
        signal = self.kp * state[2] + self.kd * state[3]
        return 1 if signal > 0 else 0


def collect_episode(
    env: CartPolePixels,
    controller: PDController,
    max_steps: int,
    expert_prob: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Run one episode with mixed expert/random actions.

    Returns a dict with `obs` of shape (T+1, 3, H, W), `actions` (T,) int,
    `dones` (T,) bool.
    """
    obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    obs_buf = [obs]
    actions: list[int] = []
    dones: list[bool] = []
    for _ in range(max_steps):
        if rng.random() < expert_prob:
            action = controller.act(env.unwrapped_state())
        else:
            action = int(rng.integers(0, 2))
        next_obs, _reward, done = env.step(action)
        obs_buf.append(next_obs)
        actions.append(action)
        dones.append(done)
        if done:
            break
    return {
        "obs": np.stack(obs_buf, axis=0),
        "actions": np.asarray(actions, dtype=np.int64),
        "dones": np.asarray(dones, dtype=bool),
    }
```

- [ ] **Step 4: Run test, verify it passes**

```bash
pytest tests/test_env.py -v
```

Expected: PASS for all four tests. If gymnasium's render returns `None`, ensure `render_mode="rgb_array"` is set (it is in this implementation).

- [ ] **Step 5: Commit**

```bash
git add src/env.py tests/test_env.py
git commit -m "feat: add CartPole pixel wrapper, PD controller, and episode collector"
```

---

### Task 5: Sequence Replay Buffer (TDD)

**Files:**
- Test: `tests/test_buffer.py`
- Create: `src/buffer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_buffer.py`:

```python
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_buffer.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/buffer.py`**

```python
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
```

- [ ] **Step 4: Run test, verify it passes**

```bash
pytest tests/test_buffer.py -v
```

Expected: PASS for all four tests.

- [ ] **Step 5: Commit**

```bash
git add src/buffer.py tests/test_buffer.py
git commit -m "feat: add SequenceReplayBuffer with chunked episode sampling"
```

---

### Task 6: WorldModel Glue & Loss (TDD)

**Files:**
- Test: `tests/test_world_model.py`
- Create: `src/world_model.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_world_model.py`:

```python
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
pytest tests/test_world_model.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `src/world_model.py`**

```python
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
```

- [ ] **Step 4: Run test, verify it passes**

```bash
pytest tests/test_world_model.py -v
```

Expected: PASS for all three tests. If `test_one_optim_step_decreases_loss` fails because the KL clamp pins it at the floor, increase the test obs scale or run more steps; the recon term should dominate on random data.

- [ ] **Step 5: Commit**

```bash
git add src/world_model.py tests/test_world_model.py
git commit -m "feat: add WorldModel glue with reconstruction + KL loss"
```

---

### Task 7: Visualization Helpers

**Files:**
- Create: `src/viz.py`

No tests — these are pure I/O helpers exercised end-to-end by `eval.py`.

- [ ] **Step 1: Implement `src/viz.py`**

```python
"""Video and figure helpers for posterior reconstruction, imagined rollouts, and latent PCA."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skvideo.io
from sklearn.decomposition import PCA


def _to_uint8(frame: np.ndarray) -> np.ndarray:
    """frame: (3, H, W) float in [0,1] -> (H, W, 3) uint8"""
    f = np.clip(frame, 0.0, 1.0)
    f = np.transpose(f, (1, 2, 0))
    return (f * 255.0).astype(np.uint8)


def write_side_by_side_video(
    truth: np.ndarray,
    pred: np.ndarray,
    path: str,
    fps: int = 15,
) -> None:
    """truth, pred: (T, 3, H, W) float arrays in [0, 1]."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    T = truth.shape[0]
    frames = []
    for t in range(T):
        left = _to_uint8(truth[t])
        right = _to_uint8(pred[t])
        gap = np.zeros((left.shape[0], 4, 3), dtype=np.uint8)
        frames.append(np.concatenate([left, gap, right], axis=1))
    video = np.stack(frames, axis=0)
    skvideo.io.vwrite(path, video, inputdict={"-r": str(fps)}, outputdict={"-r": str(fps)})


def plot_latent_pca(
    features: np.ndarray,
    color_values: np.ndarray,
    path: str,
    title: str = "Latent space (PCA, colored by pole angle)",
) -> None:
    """features: (N, D); color_values: (N,) — typically pole angle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=color_values, cmap="viridis", s=4)
    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    fig.colorbar(sc, ax=ax, label="pole angle (rad)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 2: Smoke-test the helpers**

```bash
python -c "
import numpy as np
from src.viz import write_side_by_side_video, plot_latent_pca
write_side_by_side_video(np.random.rand(10,3,64,64).astype(np.float32),
                          np.random.rand(10,3,64,64).astype(np.float32),
                          'video/posterior/_smoke.mp4')
plot_latent_pca(np.random.rand(200, 230).astype(np.float32),
                np.random.uniform(-0.2, 0.2, size=200),
                'figures/_smoke.png')
print('viz smoke OK')
"
```

Expected: prints `viz smoke OK`. Both files exist on disk afterward; remove them.

```bash
rm video/posterior/_smoke.mp4 figures/_smoke.png
```

- [ ] **Step 3: Commit**

```bash
git add src/viz.py
git commit -m "feat: add side-by-side video and latent PCA helpers"
```

---

### Task 8: Training Entry Point

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: Implement `src/train.py`**

```python
"""Training entry point.

Usage:
    python -m src.train --config config.yaml
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import yaml

from src.buffer import SequenceReplayBuffer
from src.env import CartPolePixels, PDController, collect_episode
from src.viz import write_side_by_side_video
from src.world_model import WorldModel


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(np_batch: dict, device: torch.device) -> dict:
    return {
        "obs": torch.from_numpy(np_batch["obs"]).to(device),
        "actions": torch.from_numpy(np_batch["actions"]).to(device),
        "dones": torch.from_numpy(np_batch["dones"]).to(device),
    }


def _write_train_videos(
    wm: WorldModel,
    env: CartPolePixels,
    controller: PDController,
    rng: np.random.Generator,
    step: int,
    cfg: dict,
    device: torch.device,
) -> None:
    """Collect one expert episode and emit posterior + imagined rollout mp4s."""
    ep = collect_episode(env, controller, max_steps=cfg["env"]["max_steps"], expert_prob=1.0, rng=rng)
    seq_len = cfg["buffer"]["seq_len"]
    if ep["actions"].shape[0] < seq_len:
        return
    obs = torch.from_numpy(ep["obs"][: seq_len + 1]).unsqueeze(0).to(device)
    a_idx = ep["actions"][:seq_len]
    actions = torch.zeros(1, seq_len, cfg["model"]["action_dim"], device=device)
    actions[0, np.arange(seq_len), a_idx] = 1.0

    wm.eval()
    with torch.no_grad():
        h_seq, z_seq, _, _ = wm.observe(obs[:, :seq_len], actions)
        recon = wm.decode(h_seq, z_seq).cpu().numpy()[0]
    wm.train()

    truth = ep["obs"][:seq_len]
    out_path = os.path.join(cfg["train"]["video_dir"], "posterior", f"step_{step:06d}.mp4")
    write_side_by_side_video(truth, recon, out_path)


def train(cfg: dict) -> None:
    device = _device()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    os.makedirs(cfg["train"]["model_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["train"]["video_dir"], "posterior"), exist_ok=True)
    os.makedirs(os.path.join(cfg["train"]["video_dir"], "rollouts"), exist_ok=True)

    env = CartPolePixels(image_size=cfg["env"]["image_size"])
    controller = PDController(kp=cfg["env"]["pd_kp"], kd=cfg["env"]["pd_kd"])
    buffer = SequenceReplayBuffer(
        capacity=cfg["buffer"]["capacity"], action_dim=cfg["model"]["action_dim"]
    )

    print(f"Pre-filling buffer with {cfg['buffer']['init_episodes']} episodes...")
    for _ in range(cfg["buffer"]["init_episodes"]):
        ep = collect_episode(
            env,
            controller,
            max_steps=cfg["env"]["max_steps"],
            expert_prob=cfg["env"]["expert_prob"],
            rng=rng,
        )
        buffer.add(ep)

    wm = WorldModel(
        embed_dim=cfg["model"]["embed_dim"],
        h_dim=cfg["model"]["h_dim"],
        z_dim=cfg["model"]["z_dim"],
        action_dim=cfg["model"]["action_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        min_std=cfg["model"]["min_std"],
        kl_weight=cfg["train"]["kl_weight"],
        free_nats=cfg["train"]["free_nats"],
    ).to(device)
    optim = torch.optim.Adam(wm.parameters(), lr=cfg["train"]["lr"], eps=cfg["train"]["adam_eps"])

    t0 = time.time()
    for step in range(1, cfg["train"]["train_steps"] + 1):
        np_batch = buffer.sample(
            batch_size=cfg["train"]["batch_size"],
            seq_len=cfg["buffer"]["seq_len"],
            rng=rng,
        )
        batch = _to_tensor(np_batch, device)
        out = wm.loss(batch["obs"], batch["actions"])
        optim.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg["train"]["grad_clip"])
        optim.step()

        if step % cfg["train"]["collect_every"] == 0:
            ep = collect_episode(
                env,
                controller,
                max_steps=cfg["env"]["max_steps"],
                expert_prob=cfg["env"]["expert_prob"],
                rng=rng,
            )
            buffer.add(ep)

        if step % cfg["train"]["log_every"] == 0:
            elapsed = time.time() - t0
            print(
                f"step {step:6d}/{cfg['train']['train_steps']} | "
                f"loss {out['loss'].item():.4f} | "
                f"recon {out['recon_loss'].item():.4f} | "
                f"kl {out['kl_loss'].item():.4f} | "
                f"buf {len(buffer)} | "
                f"{elapsed:.0f}s"
            )

        if step % cfg["train"]["checkpoint_every"] == 0:
            latest = os.path.join(cfg["train"]["model_dir"], "world_model.pt")
            snap = os.path.join(cfg["train"]["model_dir"], f"world_model_step_{step}.pt")
            torch.save(wm.state_dict(), latest)
            torch.save(wm.state_dict(), snap)

        if step % cfg["train"]["video_every"] == 0:
            _write_train_videos(wm, env, controller, rng, step, cfg, device)

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RSSM world model on CartPole pixels")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with a tiny override**

Run a 50-step training to confirm the whole pipeline works end-to-end:

```bash
python -c "
import yaml
with open('config.yaml') as f: cfg = yaml.safe_load(f)
cfg['train']['train_steps'] = 50
cfg['train']['log_every'] = 10
cfg['train']['checkpoint_every'] = 50
cfg['train']['video_every'] = 50
cfg['buffer']['init_episodes'] = 5
from src.train import train
train(cfg)
"
```

Expected: prints log lines every 10 steps, terminates cleanly, leaves `model/world_model.pt` and at least one `video/posterior/step_*.mp4`.

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add training entry point with periodic videos and checkpoints"
```

---

### Task 9: Evaluation Entry Point

**Files:**
- Create: `src/eval.py`

- [ ] **Step 1: Implement `src/eval.py`**

```python
"""Evaluation entry point.

Usage:
    python -m src.eval --config config.yaml --checkpoint model/world_model.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from src.env import CartPolePixels, PDController, collect_episode
from src.viz import plot_latent_pca, write_side_by_side_video
from src.world_model import WorldModel


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _episode_to_tensors(ep: dict, action_dim: int, device: torch.device):
    T = ep["actions"].shape[0]
    obs = torch.from_numpy(ep["obs"]).unsqueeze(0).to(device)  # (1, T+1, ...)
    a_idx = ep["actions"]
    actions = torch.zeros(1, T, action_dim, device=device)
    actions[0, np.arange(T), a_idx] = 1.0
    return obs, actions


def evaluate(cfg: dict, ckpt_path: str) -> None:
    device = _device()
    rng = np.random.default_rng(42)

    posterior_dir = os.path.join(cfg["eval"]["video_dir"], "posterior")
    rollout_dir = os.path.join(cfg["eval"]["video_dir"], "rollouts")
    figure_dir = cfg["eval"]["figure_dir"]
    os.makedirs(posterior_dir, exist_ok=True)
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    env = CartPolePixels(image_size=cfg["env"]["image_size"])
    controller = PDController(kp=cfg["env"]["pd_kp"], kd=cfg["env"]["pd_kd"])

    wm = WorldModel(
        embed_dim=cfg["model"]["embed_dim"],
        h_dim=cfg["model"]["h_dim"],
        z_dim=cfg["model"]["z_dim"],
        action_dim=cfg["model"]["action_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        min_std=cfg["model"]["min_std"],
        kl_weight=cfg["train"]["kl_weight"],
        free_nats=cfg["train"]["free_nats"],
    ).to(device)
    wm.load_state_dict(torch.load(ckpt_path, map_location=device))
    wm.eval()

    context_len = cfg["eval"]["context_len"]
    horizon = cfg["eval"]["imagine_horizon"]
    total_len = context_len + horizon

    all_features: list[np.ndarray] = []
    all_angles: list[np.ndarray] = []

    for i in range(cfg["eval"]["num_episodes"]):
        ep = collect_episode(
            env,
            controller,
            max_steps=cfg["env"]["max_steps"],
            expert_prob=1.0,
            rng=rng,
        )
        if ep["actions"].shape[0] < total_len:
            continue

        obs, actions = _episode_to_tensors(ep, cfg["model"]["action_dim"], device)

        # 1. Posterior reconstruction over the full sliced length
        with torch.no_grad():
            h_seq, z_seq, _, _ = wm.observe(obs[:, :total_len], actions[:, :total_len])
            recon = wm.decode(h_seq, z_seq).cpu().numpy()[0]
        truth = ep["obs"][:total_len]
        write_side_by_side_video(
            truth, recon, os.path.join(posterior_dir, f"episode_{i}.mp4")
        )

        # 2. Imagined rollout: condition on first context_len steps, then imagine.
        with torch.no_grad():
            h_ctx, z_ctx, _, _ = wm.observe(
                obs[:, :context_len], actions[:, :context_len]
            )
            init_state = (h_ctx[:, -1], z_ctx[:, -1])
            future_actions = actions[:, context_len:total_len]
            h_im, z_im = wm.imagine(init_state, future_actions)
            imagined = wm.decode(h_im, z_im).cpu().numpy()[0]
        truth_future = ep["obs"][context_len:total_len]
        write_side_by_side_video(
            truth_future, imagined, os.path.join(rollout_dir, f"episode_{i}.mp4")
        )

        # 3. Collect features for PCA: use posterior (h, z) across whole episode.
        feats = torch.cat([h_seq[0], z_seq[0]], dim=-1).cpu().numpy()  # (T, h+z)
        all_features.append(feats)
        # Underlying state isn't recorded per-step in collect_episode; approximate
        # by re-running the environment with the same actions to get pole angles.
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        angles = []
        for t in range(total_len):
            angles.append(env.unwrapped_state()[2])
            env.step(int(ep["actions"][t]))
        all_angles.append(np.asarray(angles, dtype=np.float32))

    if all_features:
        feats = np.concatenate(all_features, axis=0)
        angles = np.concatenate(all_angles, axis=0)
        plot_latent_pca(feats, angles, os.path.join(figure_dir, "latent_pca.png"))

    env.close()
    print("Evaluation complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ckpt = args.checkpoint or cfg["eval"]["checkpoint"]
    evaluate(cfg, ckpt)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test eval against the checkpoint produced in Task 8**

```bash
python -m src.eval --config config.yaml --checkpoint model/world_model.pt
```

Expected: prints `Evaluation complete.` and creates `video/posterior/episode_*.mp4`, `video/rollouts/episode_*.mp4`, `figures/latent_pca.png`. Visual quality at this point is poor (only 50 train steps); we just need the pipeline to run.

- [ ] **Step 3: Commit**

```bash
git add src/eval.py
git commit -m "feat: add evaluation entry point producing reconstruction, rollout, PCA artifacts"
```

---

### Task 10: Conceptual Companion Doc

**Files:**
- Create: `docs/rssm_explained.md`

- [ ] **Step 1: Write `docs/rssm_explained.md`**

```markdown
# The Recurrent State Space Model

## The Core Idea

A world model takes pixel observations and learns a compact latent state that lets it *predict the future*. The RSSM (Hafner et al. 2019) splits this latent into two parts that propagate together through time:

- A **deterministic** GRU hidden state `h_t` — carries history.
- A **stochastic** Gaussian latent `z_t` — captures what's uncertain about a single step.

## The Two Distributions

For each timestep, the RSSM defines two distributions over `z_t`:

| Name | Conditioned on | When used |
|---|---|---|
| **Prior** `p(z_t \| h_t)` | only the previous state | imagining without observations |
| **Posterior** `q(z_t \| h_t, e_t)` | previous state + observation embedding | training |

## Why Both?

- The **posterior** sees the observation, so it can reconstruct it accurately. That's the easy part.
- The **prior** does *not* see the observation. To be useful, it has to learn to predict where the posterior would land — without cheating.

The training loss enforces this directly:

```
L = ||decode(h, z_post) - obs||² + KL(posterior ‖ prior)
```

The KL term penalizes the prior for diverging from the posterior. Over training, the prior becomes a learned model of "what `z` should look like given `h`" — i.e., a learned dynamics function in latent space.

## What "Imagination" Means

Once trained, you can do this:

1. Encode 5 real frames into context state `(h_5, z_5)` using the posterior.
2. From step 6 onward, *only use the prior* — sample `z_t ~ p(z_t | h_t)`.
3. Decode each `(h_t, z_t)` into an image.

What you see is the model "imagining" forward without sensory input. This is exactly what Dreamer uses to train an actor-critic — the agent practices in the dreamed-up rollouts.

## Where It Breaks Down

The KL only enforces a soft match. As you imagine more steps, tiny prior errors compound. After ~20-50 steps on CartPole, imagined trajectories visibly drift — the cart teleports, the pole goes through the cart, etc. Watching the breakdown is the whole point of Level 1.

## The Shape of `(h, z)` (this project)

| Symbol | Dim | Role |
|---|---|---|
| `h_t` | 200 | GRU hidden state — accumulates history |
| `z_t` | 30 | Gaussian sample — per-step stochasticity |
| `e_t` | 1024 | Encoder output of `o_t` (observation embedding) |
| `a_t` | 2 | One-hot action |

The decoder takes `[h_t, z_t]` (concatenated, 230-dim) and reconstructs the 64×64 RGB observation.
```

- [ ] **Step 2: Commit**

```bash
git add docs/rssm_explained.md
git commit -m "docs: add plain-English RSSM companion explainer"
```

---

### Task 11: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Tiny RSSM World Model on CartPole

A minimal Dreamer-style Recurrent State Space Model trained to predict pixel observations of CartPole. Built to *understand the mechanics* of a world model: encode pixels into a recurrent latent, propagate it through time, "imagine" forward without observations, and watch the imagination break down.

***Summary***
1. [Mechanics](#1-mechanics)
2. [Approach](#2-approach)
3. [Results](#3-results)

See [`docs/rssm_explained.md`](docs/rssm_explained.md) for a plain-English walkthrough of the RSSM itself.

## 1. Mechanics

- Environment: `CartPole-v1`, observed as 64×64 RGB renders. The agent's reward is ignored (this is a world model, not a policy).
- Trajectories are collected with a mix: 50% random actions, 50% from a hand-tuned PD controller (`kp=10, kd=1`) that balances the pole indefinitely. Mixed data gives the model both "boring success" and "failure modes" in the same buffer.
- Observations are stored as whole episodes; the world model is trained on contiguous 50-step chunks.

## 2. Approach

The RSSM splits its latent into a **deterministic** GRU hidden state `h_t` and a **stochastic** Gaussian latent `z_t`. Two distributions are defined per step:

- **Prior** `p(z_t | h_t)` — what the model imagines without seeing the observation.
- **Posterior** `q(z_t | h_t, e_t)` — what the model believes after encoding the observation `o_t` to `e_t`.

The training objective is

```
L = ||decode(h, z_post) − obs||²  +  KL(posterior ‖ prior)
```

The KL term forces the prior to learn to match the posterior — that's where the imagination capability comes from.

```
              ┌───────────┐         ┌───────────┐
   o_t ──▶    │  Encoder  │ ──e_t──▶│ Posterior │── z_t (q)
              └───────────┘         └────┬──────┘
                                         │
                                  ┌──────▼──────┐
   z_{t-1}, a_{t-1} ──▶ GRU ───── │   prior     │── ẑ_t (p)
                          h_t     └──────┬──────┘
                                         │
                          ┌──────────────▼──────────────┐
                          │  Decoder([h_t, z_t]) ── ô_t │
                          └─────────────────────────────┘
```

## 3. Results

Three artifacts produced by `python -m src.eval`:

- **`video/posterior/episode_*.mp4`** — ground truth | posterior reconstruction. Sanity check: can the model encode and decode at all?
- **`video/rollouts/episode_*.mp4`** — ground truth | imagination. The first 5 frames condition the posterior; from step 6 onward the prior runs alone. Watch the imagination diverge from reality after ~10-30 steps.
- **`figures/latent_pca.png`** — 2D PCA of `(h, z)` features colored by pole angle. A visible color gradient means the latent space organized itself meaningfully.

> Result gifs/figures will be added here once the model is trained.

## Usage

```bash
pip install -r requirements.txt
python -m src.train --config config.yaml
python -m src.eval --config config.yaml --checkpoint model/world_model.pt
pytest                # run all tests
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with mechanics, approach, and results sections"
```

---

### Task 12: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
pytest -v
```

Expected: every test in `tests/` passes (encoder + rssm + env + buffer + world_model).

- [ ] **Step 2: Run a 200-step training smoke check**

```bash
python -c "
import yaml
with open('config.yaml') as f: cfg = yaml.safe_load(f)
cfg['train']['train_steps'] = 200
cfg['train']['log_every'] = 50
cfg['train']['checkpoint_every'] = 200
cfg['train']['video_every'] = 200
cfg['buffer']['init_episodes'] = 10
from src.train import train
train(cfg)
"
```

Expected: terminates cleanly, log lines show `loss` *decreasing* across the 200 steps (it should drop noticeably even in this short window).

- [ ] **Step 3: Run eval against the smoke checkpoint**

```bash
python -m src.eval --config config.yaml --checkpoint model/world_model.pt
```

Expected: produces all three artifact types (posterior mp4s, rollout mp4s, PCA png) without error. Visual quality will be poor; that's fine — quality is the job of the real 20k-step training run.

- [ ] **Step 4: Commit final state**

```bash
git status   # should be clean (or only ignored files dirty)
```

If anything's outstanding, commit it. Otherwise the project is ready for the full training run:

```bash
python -m src.train --config config.yaml
```

---

## Done

After Task 12, you have:
- 5 green test files
- A trainable world model with a working CLI
- Posterior + imagined-rollout videos as the visible "imagination breakdown" demo
- A latent PCA plot
- A README and conceptual companion doc, both in the rat-cheese project's documentation style

The next experiment (Level 2) would add a reward head and use the imagination capability for actor-critic — but that's a separate spec.
