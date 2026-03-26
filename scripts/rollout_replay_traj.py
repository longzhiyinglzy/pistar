from __future__ import annotations

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import einops
import mediapy
import numpy as np
import torch
from accelerate import Accelerator

from openpi.models.ctrl_world import CtrlWorld
from openpi.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from openpi.training.config_wm import wm_args


# GT-conditioned long-horizon replay.
# This script evaluates world model only (no policy, no dynamics, no raw-time mapping).


def merge_args(cfg, cli_args):
    for k, v in vars(cli_args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def build_parser() -> ArgumentParser:
    # Parser only provides optional overrides for config_wm.py defaults.
    parser = ArgumentParser()

    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument("--dataset_cfgs", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--replay_output_dir", type=str, default=None)

    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--max_replay_steps", type=int, default=None)
    parser.add_argument("--save_info", type=lambda x: str(x).lower() in {"1", "true", "t", "yes", "y"}, default=None)

    return parser


def build_replay_window_indices(
    current_t: int,
    buffer_start_t: int,
    cache_len: int,
    history_relative_offsets: list[int],
) -> tuple[list[int], list[int]]:
    """Build history indices for replay buffer and GT timeline.

    History semantic on 5Hz timeline is fixed as:
      [oldest, t-12, t-9, t-6, t-4, t-2, t-1]
    where oldest is the oldest available frame in current replay buffer.
    """
    oldest = max(buffer_start_t, current_t - int(cache_len) + 1)
    history_global = [oldest] + [max(oldest, current_t + int(o)) for o in history_relative_offsets]
    history_local = [g - buffer_start_t for g in history_global]
    return history_global, history_local


class ReplayAgent:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        ckpt_path = getattr(args, "ckpt_path", None)
        if not ckpt_path or not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Invalid --ckpt_path: {ckpt_path}")

        self.model = CtrlWorld(args)
        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
        self.model.to(self.device).to(self.dtype)
        self.model.eval()
        logging.info("Loaded world model from %s", ckpt_path)

        with open(args.data_stat_path, "r") as f:
            stat = json.load(f)
        if "wm_state_01" not in stat or "wm_state_99" not in stat:
            raise ValueError(f"Missing wm_state_01/wm_state_99 in {args.data_stat_path}")

        self.action_p01 = np.asarray(stat["wm_state_01"], dtype=np.float32)[None, :]
        self.action_p99 = np.asarray(stat["wm_state_99"], dtype=np.float32)[None, :]
        if self.action_p01.shape[-1] != 7 or self.action_p99.shape[-1] != 7:
            raise ValueError(
                f"Expected 7D wm_state stats, got {self.action_p01.shape[-1]}/{self.action_p99.shape[-1]}"
            )

    @staticmethod
    def normalize_bound(
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def forward_wm(
        self,
        action_cond: np.ndarray,
        current_latent: torch.Tensor,
        his_cond: torch.Tensor,
        text: Optional[str],
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Forward world model on one 5Hz window.

        Returns:
            pred_rgb_views: uint8 array [3, F, H, W, 3]
            pred_latents_merged: tensor [F, 4, 72, 40]
        """
        args = self.args

        pose6 = np.asarray(action_cond[:, :6], dtype=np.float32)
        gripper = np.asarray(action_cond[:, 6:7], dtype=np.float32)
        p01_6 = np.asarray(self.action_p01[:, :6], dtype=np.float32)
        p99_6 = np.asarray(self.action_p99[:, :6], dtype=np.float32)

        # Keep WM conditioning semantics aligned with training/rollout:
        # normalize pose6 only, gripper passthrough.
        pose6_norm = self.normalize_bound(pose6, p01_6, p99_6, clip_min=-1, clip_max=1)
        action_cond_norm = np.concatenate([pose6_norm, gripper], axis=-1)

        action_cond_t = torch.tensor(action_cond_norm, dtype=self.dtype, device=self.device).unsqueeze(0)
        current_latent = current_latent.to(self.device, dtype=self.dtype)
        his_cond = his_cond.to(self.device, dtype=self.dtype)

        if text:
            text_token = self.model.action_encoder(
                action_cond_t,
                [text],
                self.model.tokenizer,
                self.model.text_encoder,
                args.frame_level_cond,
            )
        else:
            text_token = self.model.action_encoder(action_cond_t)

        with torch.no_grad():
            pipeline = self.model.pipeline
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=current_latent,
                text=text_token,
                width=args.width,
                height=int(args.height * 3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type="latent",
                return_dict=False,
                frame_level_cond=args.frame_level_cond,
                his_cond_zero=args.his_cond_zero,
            )

        # [1, F, 4, 72, 40]
        latents_merged = latents[0].detach().to(torch.float32).cpu()

        # [1, F, 4, 72, 40] -> [3, F, 4, 24, 40]
        latents_views = einops.rearrange(latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)

        decoded = []
        x = latents_views.flatten(0, 1)
        decode_kwargs = {}
        with torch.no_grad():
            for i in range(0, x.shape[0], args.decode_chunk_size):
                chunk = x[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
                decode_kwargs["num_frames"] = chunk.shape[0]
                decoded.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded, dim=0)
        videos = videos.reshape(latents_views.shape[0], latents_views.shape[1], *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        return videos, latents_merged

    def decode_gt_views(self, gt_view_latents: list[torch.Tensor]) -> np.ndarray:
        """Decode full GT latent sequence for three views.

        Args:
            gt_view_latents: list of 3 tensors, each [T, 4, 24, 40]
        Returns:
            uint8 [3, T, H, W, 3]
        """
        args = self.args
        pipeline = self.model.pipeline

        outs = []
        with torch.no_grad():
            for view_lat in gt_view_latents:
                x = view_lat.to(self.device, dtype=self.dtype)
                decoded = []
                decode_kwargs = {}
                for i in range(0, x.shape[0], args.decode_chunk_size):
                    chunk = x[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
                    decode_kwargs["num_frames"] = chunk.shape[0]
                    decoded.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
                video = torch.cat(decoded, dim=0)
                video = ((video / 2.0 + 0.5).clamp(0, 1) * 255)
                video = video.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
                outs.append(video)

        return np.stack(outs, axis=0)


def _concat_three_views(videos_3tfhwc: np.ndarray) -> np.ndarray:
    """[3, T, H, W, C] -> [T, H, 3W, C]"""
    if videos_3tfhwc.shape[0] != 3:
        raise ValueError(f"Expected 3 views, got shape {videos_3tfhwc.shape}")
    return np.concatenate([videos_3tfhwc[v] for v in range(3)], axis=2)


def _load_episode(
    dataset_root: Path,
    mode: str,
    episode_id: int,
) -> tuple[dict, list[torch.Tensor], np.ndarray, str, int]:
    ann_path = dataset_root / "annotation" / mode / f"{episode_id}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    with open(ann_path, "r") as f:
        ann = json.load(f)

    pose = np.asarray(ann["abs_pose"], dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"abs_pose must be [T,7], got {pose.shape} in {ann_path}")

    video_length = int(ann["video_length"])
    text = str((ann.get("texts") or [""])[0])

    lat_paths: list[Path] = []
    if isinstance(ann.get("latent_videos"), list) and len(ann["latent_videos"]) >= 3:
        for entry in ann["latent_videos"][:3]:
            rel = entry.get("latent_video_path", "")
            lat_paths.append(dataset_root / rel)
    else:
        for i in range(3):
            lat_paths.append(dataset_root / "latent_videos" / mode / str(episode_id) / f"{i}.pt")

    latents = []
    for p in lat_paths:
        if not p.exists():
            raise FileNotFoundError(f"Latent file not found: {p}")
        x = torch.load(p, map_location="cpu")
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(torch.float32)
        latents.append(x)

    for i, x in enumerate(latents):
        if x.ndim != 4 or x.shape[1:] != (4, 24, 40):
            raise ValueError(f"Latent view {i} should be [T,4,24,40], got {tuple(x.shape)} at {lat_paths[i]}")
        if x.shape[0] != video_length:
            raise ValueError(
                f"Latent length mismatch view {i}: latent T={x.shape[0]} vs video_length={video_length} in {ann_path}"
            )

    return ann, latents, pose, text, video_length


def main(args):
    mode = str(getattr(args, "mode", None) or "val")
    start_frame = int(getattr(args, "start_frame", 0) if getattr(args, "start_frame", None) is not None else 0)
    max_replay_steps = getattr(args, "max_replay_steps", None)
    if max_replay_steps is not None:
        max_replay_steps = int(max_replay_steps)
        if max_replay_steps <= 0:
            raise ValueError("max_replay_steps must be positive")

    dataset_names = [x for x in str(args.dataset_names).split("+") if x]
    if len(dataset_names) == 0:
        raise ValueError("dataset_names is empty")
    dataset_name = dataset_names[0]
    dataset_root = Path(args.dataset_root_path) / dataset_name

    episode_id = int(args.episode_id)

    if len(getattr(args, "history_relative_offsets", [])) != int(args.num_history) - 1:
        raise ValueError("history_relative_offsets length must equal num_history-1")

    agent = ReplayAgent(args)

    _, gt_view_latents, gt_pose, text, video_length = _load_episode(dataset_root, mode, episode_id)
    if start_frame < 0 or start_frame >= video_length:
        raise ValueError(f"start_frame out of range: {start_frame}, video_length={video_length}")

    # Decode full GT sequence once, then slice by frame id for long-sequence visualization.
    gt_views_rgb_all = agent.decode_gt_views(gt_view_latents)  # [3, T, H, W, 3]
    gt_frames_cat_all = _concat_three_views(gt_views_rgb_all)  # [T, H, 3W, 3]

    # Build merged latent sequence [T, 4, 72, 40] from 3-view latent episodes.
    gt_merged = torch.cat(gt_view_latents, dim=2)  # concat on latent-height: 24*3=72

    cache_len = int(args.wm_history_cache_len)
    replay_stride = int(args.num_frames) - 1
    if replay_stride <= 0:
        raise ValueError("num_frames must be >= 2 for long-horizon replay")

    # Initialize replay buffers with frame-0 repeat, then warm up to start_frame using GT.
    frame0_lat = gt_merged[0].clone()
    frame0_pose = gt_pose[0].copy()
    latent_buffer: list[torch.Tensor] = [frame0_lat.clone() for _ in range(cache_len)]
    pose_buffer: list[np.ndarray] = [frame0_pose.copy() for _ in range(cache_len)]

    buffer_start_t = 0
    for g in range(1, start_frame + 1):
        latent_buffer.append(gt_merged[g].clone())
        pose_buffer.append(gt_pose[g].copy())
        if len(latent_buffer) > cache_len:
            latent_buffer = latent_buffer[-cache_len:]
            pose_buffer = pose_buffer[-cache_len:]
            buffer_start_t = g - len(latent_buffer) + 1

    current_t = start_frame

    gt_long_frames: list[np.ndarray] = [gt_frames_cat_all[start_frame]]
    replay_long_frames: list[np.ndarray] = [gt_frames_cat_all[start_frame]]

    num_replay_steps = 0
    stop_reason = "future_insufficient"

    while True:
        if max_replay_steps is not None and num_replay_steps >= max_replay_steps:
            stop_reason = "max_replay_steps_reached"
            break

        if current_t + int(args.num_frames) - 1 >= video_length:
            stop_reason = "future_insufficient"
            break

        history_global, history_local = build_replay_window_indices(
            current_t=current_t,
            buffer_start_t=buffer_start_t,
            cache_len=cache_len,
            history_relative_offsets=list(args.history_relative_offsets),
        )
        future_global = [current_t + i for i in range(int(args.num_frames))]

        # Build WM inputs.
        his_cond = torch.stack([latent_buffer[idx] for idx in history_local], dim=0).unsqueeze(0)
        current_latent = latent_buffer[-1].unsqueeze(0)

        his_pose = gt_pose[np.asarray(history_global, dtype=np.int64)]
        future_pose = gt_pose[np.asarray(future_global, dtype=np.int64)]
        action_cond = np.concatenate([his_pose, future_pose], axis=0).astype(np.float32)

        pred_views_rgb, pred_latents_merged = agent.forward_wm(
            action_cond=action_cond,
            current_latent=current_latent,
            his_cond=his_cond,
            text=text if bool(args.text_cond) else None,
        )

        pred_frames_cat = _concat_three_views(pred_views_rgb)  # [F, H, 3W, 3]

        # Buffer update: append only future 4 frames (t+1..t+4).
        for j in range(1, int(args.num_frames)):
            g = future_global[j]
            latent_buffer.append(pred_latents_merged[j].clone())
            pose_buffer.append(gt_pose[g].copy())
            gt_long_frames.append(gt_frames_cat_all[g])
            replay_long_frames.append(pred_frames_cat[j])

        if len(latent_buffer) > cache_len:
            latent_buffer = latent_buffer[-cache_len:]
            pose_buffer = pose_buffer[-cache_len:]

        buffer_start_t = current_t + replay_stride - len(latent_buffer) + 1
        current_t += replay_stride
        num_replay_steps += 1

    gt_long = np.stack(gt_long_frames, axis=0)
    replay_long = np.stack(replay_long_frames, axis=0)
    vis = np.concatenate([gt_long, replay_long], axis=1)  # vertical stack: top GT, bottom replay

    out_dir = Path(args.replay_output_dir) / "replay_traj"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"episode_{episode_id}.mp4"
    out_json = out_dir / f"episode_{episode_id}.json"

    mediapy.write_video(str(out_video), vis, fps=int(args.fps))

    info = {
        "episode_id": int(episode_id),
        "mode": mode,
        "text": text,
        "video_length": int(video_length),
        "start_frame": int(start_frame),
        "num_history": int(args.num_history),
        "num_frames": int(args.num_frames),
        "wm_history_cache_len": int(args.wm_history_cache_len),
        "num_replay_steps": int(num_replay_steps),
        "replay_stride": int(replay_stride),
        "stop_reason": stop_reason,
    }

    if bool(getattr(args, "save_info", True)):
        with open(out_json, "w") as f:
            json.dump(info, f, indent=2)

    logging.info("Saved replay video: %s", out_video)
    if bool(getattr(args, "save_info", True)):
        logging.info("Saved replay metadata: %s", out_json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = build_parser()
    cli_args = parser.parse_args()

    args = wm_args()
    args = merge_args(args, cli_args)
    args.__post_init__()

    main(args)
