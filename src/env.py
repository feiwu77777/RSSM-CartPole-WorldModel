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
