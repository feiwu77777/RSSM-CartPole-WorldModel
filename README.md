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
