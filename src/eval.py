"""Evaluation entry point.

Usage:
    python -m src.eval --config config.yaml --checkpoint model/world_model.pt
"""

from __future__ import annotations

import argparse
import os

# Configure ffmpeg path for skvideo before any skvideo import occurs.
# If ffmpeg/ffprobe are not on PATH, fall back to the imageio-ffmpeg bundled binary
# by creating temporary symlinks that skvideo can discover.
def _ensure_ffmpeg() -> None:
    import shutil
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return  # already available
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_dir = os.path.join(os.path.dirname(ffmpeg_exe), "_skvideo_links")
        os.makedirs(tmp_dir, exist_ok=True)
        for name in ("ffmpeg", "ffprobe"):
            link = os.path.join(tmp_dir, name)
            if not os.path.exists(link):
                os.symlink(ffmpeg_exe, link)
        import skvideo
        skvideo.setFFmpegPath(tmp_dir)
    except Exception:
        pass  # best-effort; video writing will raise if ffmpeg is truly absent

_ensure_ffmpeg()

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
