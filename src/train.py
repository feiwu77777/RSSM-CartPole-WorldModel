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
