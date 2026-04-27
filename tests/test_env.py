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
