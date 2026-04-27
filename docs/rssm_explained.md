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
