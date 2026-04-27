# Tiny RSSM World Model on CartPole — Design

**Date:** 2026-04-27
**Status:** Approved

## Goal

Build a minimal Recurrent State Space Model (RSSM) — the core of Dreamer — and train it on pixel observations of CartPole. The objective is **understanding the mechanics of a world model**, not control: train a model that can encode pixel observations, propagate a recurrent latent state through time, and "imagine" forward without further observations.

This is Level 1 of a larger world-model exploration: build the smallest possible RSSM, look at what it learns, and watch its imagination break down over long horizons.

---

## Scope

**In scope**
- Pixel-based observations (64×64 RGB) from CartPole-v1
- Full Dreamer-style RSSM: deterministic GRU state `h` + stochastic Gaussian latent `z`, with prior `p(z|h)` and posterior `q(z|h, e)`
- Training objective: pixel reconstruction (MSE) + KL(posterior ‖ prior) with free-nats clamp
- Mixed data collection: PD controller (expert) + random actions
- Three evaluation artifacts: posterior reconstruction videos, imagined-rollout videos, latent-space PCA plot

**Out of scope (Level 2+)**
- Reward prediction head
- Imagination-based policy / actor-critic
- Continuous-action environments

---

## File Structure

```
RSSM-CartPole-WorldModel/
├── src/
│   ├── __init__.py
│   ├── encoder.py       # CNN encoder + ConvTranspose decoder
│   ├── rssm.py          # RSSM: GRU + prior + posterior
│   ├── world_model.py   # WorldModel: encoder + RSSM + decoder + loss
│   ├── env.py           # CartPolePixels + PDController + collect_episode
│   ├── buffer.py        # SequenceReplayBuffer (whole episodes, chunked sampling)
│   ├── train.py         # CLI training entry point
│   ├── eval.py          # CLI evaluation entry point
│   └── viz.py           # mp4 + matplotlib helpers shared by train/eval
├── tests/
│   ├── __init__.py
│   ├── test_encoder.py
│   ├── test_rssm.py
│   ├── test_buffer.py
│   ├── test_env.py
│   └── test_world_model.py
├── config.yaml
├── model/                  # checkpoints (.pt)
├── video/
│   ├── posterior/          # observation-conditioned reconstruction
│   └── rollouts/           # ground truth | imagined side-by-side
├── figures/                # latent-space PCA plots
├── docs/
│   ├── superpowers/specs/
│   ├── superpowers/plans/
│   └── rssm_explained.md   # conceptual companion (analogue of bellman_equation.md)
├── requirements.txt
└── README.md
```

---

## Section 1: `encoder.py` — Encoder & Decoder

### `Encoder(nn.Module)`
- Input: `(B, 3, 64, 64)` float in `[0, 1]`
- Stack: `Conv2d(3→32, k=4, s=2)` → ReLU → `Conv2d(32→64, k=4, s=2)` → ReLU → `Conv2d(64→128, k=4, s=2)` → ReLU → `Conv2d(128→256, k=4, s=2)` → ReLU → flatten
- Output: `(B, embed_dim)` with `embed_dim=1024`
- Matches the Dreamer small-image encoder; the exact spatial flow falls out of stride-2 padding-0 convs on 64×64 input (results in 256×2×2 = 1024 features after flatten)

### `Decoder(nn.Module)`
- Input: feature `(B, h_dim + z_dim)` (concatenation of GRU state and stochastic latent)
- `Linear(h_dim+z_dim → 1024)` → reshape `(B, 1024, 1, 1)` → ConvTranspose stack mirroring encoder → `(B, 3, 64, 64)`
- Output treated as Gaussian with fixed σ=1, so reconstruction loss is plain MSE

---

## Section 2: `rssm.py` — Recurrent State Space Model

The latent state is split into two pieces propagated together:
- `h_t` — **deterministic** GRU hidden state (history aggregator)
- `z_t` — **stochastic** Gaussian sample (per-step uncertainty)

### Three sub-networks

**Recurrent forward** (advances `h`, `nn.GRUCell`):
```
h_t = GRUCell(input = [z_{t-1}, a_{t-1}], hidden = h_{t-1})
```

**Prior** `p(z_t | h_t)` — used for imagination, no observation:
```
μ_prior, σ_prior = MLP_prior(h_t)
ẑ_t ~ N(μ_prior, σ_prior)
```
σ is parameterized as `softplus(raw) + min_std` with `min_std=0.1` to prevent collapse.

**Posterior** `q(z_t | h_t, e_t)` — used during training, observation-conditioned:
```
μ_post, σ_post = MLP_post([h_t, e_t])
z_t ~ N(μ_post, σ_post)
```

All MLPs are 2-layer, hidden size 200, ELU activations (Dreamer defaults).

### `RSSM` class API
- `init_state(batch_size, device) → (h, z)` — zero tensors
- `forward_prior(h, z, a) → (h_next, z_prior, prior_dist)`
- `forward_posterior(h, z, a, e_next) → (h_next, z_post, post_dist, prior_dist)` — returns both distributions so caller can compute KL

### Dimensions (config-controlled)
- `h_dim=200`, `z_dim=30`, `action_dim=2`, `hidden_dim=200`, `embed_dim=1024`

---

## Section 3: `world_model.py` — Glue and Loss

Owns `Encoder`, `RSSM`, `Decoder`. Exposes the training objective.

### `WorldModel(nn.Module)` API
- `observe(obs_seq, action_seq, init_state) → traj` — runs **posterior** forward over a full sequence. Returns sequence of `(h, z, post_dist, prior_dist)` plus final state.
- `imagine(init_state, action_seq) → traj` — runs **prior** forward, no observations. Returns sequence of `(h, z)`.
- `decode(h, z) → image`
- `loss(obs_seq, action_seq) → dict` — full training objective.

### Loss

```
L_recon = mean ||decoder(h_t, z_post_t) − o_t||²
L_kl    = mean KL(q(z_t | h_t, e_t) ‖ p(z_t | h_t))
L_total = L_recon + β · max(L_kl, free_nats)
```

- `β = kl_weight` (config), default 1.0
- `free_nats=3.0` — clamp prevents KL collapse (straight from Dreamer)

### Returned dict
`{loss, recon_loss, kl_loss, posterior_entropy, prior_entropy}` — all logged each train step.

### Why this works (one-line)
The posterior gets to see the future observation when reconstructing, but is penalized for diverging from the prior (which can't). So the prior is forced to learn to predict the same thing without the observation — that's the imagination capability.

---

## Section 4: `env.py` — Environment Wrapper and Behavior Policy

### `CartPolePixels`
Wraps `gymnasium.make("CartPole-v1", render_mode="rgb_array")`:
- `reset() → (3, 64, 64)` float in `[0, 1]` — calls `env.reset()`, renders, `cv2.resize` to 64×64, normalize
- `step(action) → (obs, reward, done)` — same render+resize pattern. Reward is ignored by the world model in Level 1.
- `close()` — releases underlying env

### `PDController`
Non-learned expert policy. Reads underlying `[pos, vel, angle, ang_vel]` from the gym env directly (it doesn't observe pixels — only the world model does):
```python
def act(state):
    return 1 if (kp * state[2] + kd * state[3]) > 0 else 0
```
With `kp=10, kd=1` it balances CartPole indefinitely.

### `collect_episode(env, controller, max_steps, expert_prob, rng) → episode dict`
- Episode loop: at each step, with probability `expert_prob` use `controller.act(state)`, else sample uniformly from `{0, 1}`
- Returns: `{obs: (T+1, 3, 64, 64), actions: (T,), dones: (T,)}` — actions stored as integers; one-hot conversion happens in the buffer.
- Episode ends on `done` or `max_steps`.

---

## Section 5: `buffer.py` — Sequence Replay Buffer

### `SequenceReplayBuffer`
Stores whole episodes; samples fixed-length chunks for recurrent training.

- `add(episode)` — append; evict oldest when `len(episodes) > capacity` (capacity in episodes, default 200)
- `sample(batch_size, seq_len) → batch dict`:
  - Pick `batch_size` random episodes (rejection-sample if shorter than `seq_len + 1`)
  - For each, pick a random start index in `[0, len_episode − seq_len]`
  - Slice a contiguous window of length `seq_len`
  - One-hot encode actions
- Returned shapes: `obs (B, T, 3, 64, 64)`, `actions (B, T, action_dim)`, `dones (B, T)`

### Why sequence chunks not transitions
The RSSM is recurrent — gradients flow through the GRU across timesteps. Dreamer uses `seq_len=50`; we match that.

### Mixed-buffer training scheme
- Pre-fill: `init_episodes=50` episodes before training begins
- During training: every `collect_every=5` gradient steps, append one fresh episode with `expert_prob=0.5`

---

## Section 6: `train.py` — Training Entry Point

CLI: `python -m src.train --config config.yaml`

```
1. Load config (PyYAML); seed torch + numpy
2. Build env, controller, buffer, world_model, optimizer
3. Pre-fill: collect init_episodes into buffer
4. For step in range(train_steps):
     a. batch = buffer.sample(batch_size, seq_len)
     b. losses = world_model.loss(batch.obs, batch.actions)
     c. optimizer.zero_grad(); losses['loss'].backward()
     d. clip_grad_norm_(world_model.parameters(), grad_clip)
     e. optimizer.step()
     f. if step % collect_every == 0:    collect 1 fresh episode → buffer
     g. if step % log_every == 0:        print loss components
     h. if step % checkpoint_every == 0: save model_dir/world_model.pt + snapshot
     i. if step % video_every == 0:      write 1 posterior + 1 imagined rollout mp4
```

- **Optimizer:** Adam, `lr=6e-4`, `eps=1e-4`
- **Device:** auto — `cuda` if available, else `cpu`
- **Output:** `model/world_model.pt` (latest) + `model/world_model_step_{N}.pt` (snapshots every `checkpoint_every`)

---

## Section 7: `eval.py` — Evaluation Entry Point

CLI: `python -m src.eval --config config.yaml --checkpoint model/world_model.pt`

Evaluation episodes are freshly collected with `expert_prob=1.0` (PD controller only) so they reliably reach `seq_len + context_len` steps. Produces three artifacts:

### 1. Posterior reconstruction (`video/posterior/episode_{i}.mp4`)
- Run `world_model.observe(...)` on a held-out episode
- Decode each `z_post` → image
- Write side-by-side mp4: ground truth | reconstruction
- Sanity check: can the model encode/decode at all?

### 2. Imagined rollout (`video/rollouts/episode_{i}.mp4`)
- Use first `context_len=5` observed frames + their actions to build context state via posterior
- From step 6 onward: call `world_model.imagine(...)` for `imagine_horizon=45` steps with the **actual recorded actions**
- Decode each imagined `z_prior` → image
- Write side-by-side mp4: ground truth | imagination
- This is the Dreamer demo — shows when imagination diverges from reality.

### 3. Latent space PCA (`figures/latent_pca.png`)
- Run posterior on `num_episodes` episodes; collect `(h, z)` features per timestep
- 2D PCA, color points by pole angle
- Reveals whether the latent space organizes meaningfully

---

## Section 8: `config.yaml`

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

---

## Section 9: Tests

Pytest, mirroring the rat-cheese project. CPU-only, each test runs in seconds.

| File | Tests |
|---|---|
| `test_encoder.py` | `Encoder((B,3,64,64)) → (B,1024)`; `Decoder((B,h_dim+z_dim)) → (B,3,64,64)` |
| `test_rssm.py` | `forward_prior` and `forward_posterior` return correct shapes; KL between identical Gaussians ≈ 0; 50-step rollout produces no NaNs |
| `test_buffer.py` | After adding 3 episodes, `sample(batch_size=2, seq_len=10)` returns `(2, 10, ...)`; sampled chunks are contiguous within an episode |
| `test_env.py` | Observation shape `(3, 64, 64)`, dtype `float32`, values in `[0, 1]`; PD controller balances ≥100 steps |
| `test_world_model.py` | One Adam step on a small batch decreases `loss`; loss dict contains expected keys |

---

## Dependencies (`requirements.txt`)

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

`scikit-learn` is used only for PCA in `viz.py`.

---

## Conceptual Companion

`docs/rssm_explained.md` — analogue of `bellman_equation.md` from the rat-cheese project. A short, plain-English explanation of:
- What an RSSM is and why it's recurrent
- The split between deterministic `h` and stochastic `z`
- Why the prior/posterior split + KL is the source of "imagination"
- A worked toy example showing how the prior would propagate forward without observations

---

## Success Criteria (what "done" looks like)

1. `pytest` passes (all 5 test files green)
2. `python -m src.train --config config.yaml` runs to completion (~20k steps, CPU-feasible) and reconstruction loss visibly decreases
3. `python -m src.eval --config config.yaml --checkpoint model/world_model.pt` produces:
   - At least one posterior reconstruction mp4 where the rebuilt cart and pole are recognizable
   - At least one imagined-rollout mp4 that **stays coherent for ~10 steps** and then **visibly drifts** — the breakdown is the whole point
   - A latent PCA plot with visible structure (color gradient by pole angle)
4. README sections mirror the rat-cheese style: short intro, numbered sections (Mechanics → Approach → Results), embedded gifs/images of rollouts
