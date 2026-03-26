import dataclasses
import datetime
import json
import logging
import pathlib
import shutil
from argparse import ArgumentParser
from typing import Optional

import einops
import mediapy
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from openpi.models.ctrl_world import CtrlWorld
from openpi.models.dynamics import Dynamics
from openpi.models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from openpi.policies import policy_config
from openpi.training import config as config_pi
from openpi.training.config_wm import wm_args
from openpi_client import image_tools


def _extract_first6(stats: dict, key: str) -> np.ndarray:
    if key not in stats:
        raise KeyError(key)
    arr = np.asarray(stats[key], dtype=np.float32).reshape(-1)
    if arr.shape[0] < 6:
        raise ValueError(f"Stats key {key} must have at least 6 dims, got {arr.shape[0]}")
    return arr[:6]


def load_dynamics_stats(path: str) -> dict:
    """Load dynamics stats for action(6)/pose(6) normalization (canonical only)."""
    with open(path, "r") as f:
        stats = json.load(f)

    required = ["dyn_action_01", "dyn_action_99", "dyn_pose_01", "dyn_pose_99"]
    missing = [k for k in required if k not in stats]
    if missing:
        raise ValueError(f"Dynamics stat file missing canonical keys {missing}: {path}")

    return {
        "dyn_action_01": _extract_first6(stats, "dyn_action_01"),
        "dyn_action_99": _extract_first6(stats, "dyn_action_99"),
        "dyn_pose_01": _extract_first6(stats, "dyn_pose_01"),
        "dyn_pose_99": _extract_first6(stats, "dyn_pose_99"),
    }


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value}")


def _resize_interpolate(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize image with bilinear interpolation only (no aspect-ratio preserve, no padding)."""
    img = np.ascontiguousarray(image)
    t = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    t = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
    return t.cpu().numpy()


def _resize_interpolate_float(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize image with bilinear interpolation and keep float32 [0,255] (no uint8 requantization)."""
    img = np.ascontiguousarray(image)
    t = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    t = t.squeeze(0).permute(1, 2, 0).clamp(0, 255)
    return t.cpu().numpy().astype(np.float32)


def _rotvec_to_quat(rotvec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Axis-angle (rotation vector) to quaternion [x, y, z, w]."""
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < eps:
        # First-order approximation around zero rotation.
        return np.asarray([0.5 * v[0], 0.5 * v[1], 0.5 * v[2], 1.0], dtype=np.float64)
    axis = v / theta
    half = 0.5 * theta
    s = float(np.sin(half))
    c = float(np.cos(half))
    return np.asarray([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=np.float64)


def _quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < eps:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiply for [x, y, z, w]."""
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_to_rotvec(quat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Quaternion [x, y, z, w] to axis-angle (rotation vector)."""
    q = _quat_normalize(quat, eps=eps)
    # Canonical hemisphere to avoid sign flips between q and -q.
    if q[3] < 0:
        q = -q

    vec = q[:3]
    w = float(np.clip(q[3], -1.0, 1.0))
    sin_half = float(np.linalg.norm(vec))
    if sin_half < eps:
        return (2.0 * vec).astype(np.float64)

    axis = vec / sin_half
    angle = 2.0 * float(np.arctan2(sin_half, w))
    return (axis * angle).astype(np.float64)


@dataclasses.dataclass
class RolloutExportArgs:
    save_lerobot_rollout: bool = True
    rollout_repo_id: str = "ybpy/libero_wm_rollout"
    rollout_output_dir: Optional[str] = None
    rollout_overwrite: bool = False
    rollout_robot_type: str = "panda"
    rollout_fps: int = 10
    rollout_penalty_value: float = -1.0


class LiberoRolloutLeRobotWriter:
    """Write rollout trajectories to LeRobot format (same schema as examples/libero/main.py)."""

    def __init__(self, args: RolloutExportArgs, image_size: int = 224):
        try:
            from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "save_lerobot_rollout=True requires lerobot. Install dependencies first."
            ) from exc

        self._LeRobotDataset = LeRobotDataset
        self.repo_id = args.rollout_repo_id
        self.penalty_value = args.rollout_penalty_value
        self.image_size = image_size

        if args.rollout_output_dir:
            self.dataset_path = pathlib.Path(args.rollout_output_dir) / self.repo_id
        else:
            self.dataset_path = pathlib.Path(HF_LEROBOT_HOME) / self.repo_id

        if args.rollout_overwrite and self.dataset_path.exists():
            logging.warning("Removing existing rollout dataset at %s", self.dataset_path)
            shutil.rmtree(self.dataset_path)

        self.dataset = self._create_or_load_dataset(args)
        logging.info("Initialized LeRobot dataset at %s", self.dataset_path)

    def _create_or_load_dataset(self, args: RolloutExportArgs):
        if self.dataset_path.exists() and (self.dataset_path / "meta").exists():
            dataset = self._LeRobotDataset(repo_id=self.repo_id, root=self.dataset_path)
            if hasattr(dataset, "start_image_writer"):
                dataset.start_image_writer(num_processes=5, num_threads=10)
            logging.info("Appending rollouts to existing LeRobot dataset (%s episodes).", dataset.num_episodes)
            return dataset

        if self.dataset_path.exists():
            raise ValueError(
                f"Rollout output path exists but is not a valid LeRobot dataset: {self.dataset_path}. "
                "Use --rollout_overwrite true to recreate it."
            )

        return self._LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.dataset_path,
            robot_type=args.rollout_robot_type,
            fps=args.rollout_fps,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (self.image_size, self.image_size, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (self.image_size, self.image_size, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["actions"],
                },
                "intervention": {
                    "dtype": "int64",
                    "shape": (1,),
                    "names": ["intervention_flag"],
                },
                "value_label": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["value_label"],
                },
                "reward": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["reward"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    def _compute_value_labels(self, episode_length: int, success: bool) -> np.ndarray:
        if success:
            t = np.arange(episode_length, dtype=np.float32)
            value_labels = -(episode_length - 1 - t) / float(episode_length)
            return value_labels.astype(np.float32)
        return np.full((episode_length,), self.penalty_value, dtype=np.float32)

    def _compute_rewards(self, episode_length: int, success: bool) -> np.ndarray:
        # 01 reward
        # if success:
        #     rewards = np.zeros((episode_length,), dtype=np.float32)
        #     rewards[-1] = 1.0
        #     return rewards
        # return np.zeros((episode_length,), dtype=np.float32)
        rewards = np.full((episode_length,), -1.0 / float(episode_length), dtype=np.float32)
        if success:
            rewards[-1] = 0.0
        else:
            rewards[-1] = -1.0
        return rewards

    def _add_frame(self, frame: dict, task: str):
        frame_with_task = dict(frame)
        frame_with_task["task"] = task
        try:
            self.dataset.add_frame(frame_with_task)
            return
        except TypeError:
            self.dataset.add_frame(frame)

    def save_episode(self, *, steps: list[dict], task: str, success: bool) -> None:
        if not steps:
            logging.warning("Empty rollout episode, skip LeRobot save.")
            return

        value_labels = self._compute_value_labels(len(steps), success)
        rewards = self._compute_rewards(len(steps), success)

        for idx, step in enumerate(steps):
            frame = {
                "image": step["image"],
                "wrist_image": step["wrist_image"],
                "state": step["state"],
                "actions": step["actions"],
                "intervention": np.asarray([0], dtype=np.int64),
                "value_label": np.asarray([value_labels[idx]], dtype=np.float32),
                "reward": np.asarray([rewards[idx]], dtype=np.float32),
            }
            self._add_frame(frame, task=task)

        self.dataset.save_episode()


class Agent:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        config = config_pi.get_config(args.config_name)
        self.policy = policy_config.create_trained_policy(config, args.pi_ckpt)

        infer_ckpt_path = getattr(args, "ckpt_path", None)
        if not infer_ckpt_path or not pathlib.Path(infer_ckpt_path).exists():
            raise FileNotFoundError(
                "rollout inference requires a valid --ckpt_path. "
                f"Got: {infer_ckpt_path}"
            )

        self.model = CtrlWorld(args)
        self.model.load_state_dict(torch.load(infer_ckpt_path, map_location="cpu"))
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print(f"load world model success from {infer_ckpt_path}")

        # Optional learned pose dynamics model:
        # input: current abs_pose[:6] + action_chunk[:, :6]
        # output: future absolute pose chunk[:, :6] (one future pose per input action)
        self.use_dynamics = bool(getattr(args, "use_dynamics", False))
        self.dynamics: Optional[Dynamics] = None
        if self.use_dynamics:
            dyn_ckpt_path = getattr(args, "dyn_ckpt_path", None)
            if not dyn_ckpt_path:
                raise ValueError("use_dynamics=True but --dyn_ckpt_path is not provided")
            dyn_state = torch.load(dyn_ckpt_path, map_location=self.device)

            model_kwargs = {
                "pose_dim": 6,
                "action_dim": 6,
                "action_num": int(getattr(args, "dyn_action_num", int(args.action_horizon))),
                "hidden_size": int(getattr(args, "dyn_hidden_size", 512)),
                "num_layers": int(getattr(args, "dyn_num_layers", 3)),
            }
            if isinstance(dyn_state, dict) and "model_kwargs" in dyn_state:
                model_kwargs.update(dyn_state["model_kwargs"])

            self.dynamics = Dynamics(**model_kwargs).to(self.device)

            dyn_model_state = dyn_state
            if isinstance(dyn_state, dict) and "model_state_dict" in dyn_state:
                dyn_model_state = dyn_state["model_state_dict"]
            self.dynamics.load_state_dict(dyn_model_state, strict=True)

            ckpt_dyn_stats = dyn_state.get("dyn_stats") if isinstance(dyn_state, dict) else None
            if ckpt_dyn_stats is not None:
                required = ["dyn_action_01", "dyn_action_99", "dyn_pose_01", "dyn_pose_99"]
                missing = [k for k in required if k not in ckpt_dyn_stats]
                if missing:
                    raise ValueError(
                        f"Dynamics checkpoint dyn_stats missing canonical keys {missing}: {dyn_ckpt_path}"
                    )
                self.dynamics.set_normalization_stats(
                    action_min=np.asarray(ckpt_dyn_stats["dyn_action_01"], dtype=np.float32),
                    action_max=np.asarray(ckpt_dyn_stats["dyn_action_99"], dtype=np.float32),
                    pose_min=np.asarray(ckpt_dyn_stats["dyn_pose_01"], dtype=np.float32),
                    pose_max=np.asarray(ckpt_dyn_stats["dyn_pose_99"], dtype=np.float32),
                )
                logging.info("Loaded dynamics normalization stats from checkpoint: %s", dyn_ckpt_path)
            else:
                dyn_stat_path = getattr(args, "dyn_stat_path", None)
                if not dyn_stat_path:
                    raise ValueError(
                        "Dynamics stats missing in checkpoint. Please provide --dyn_stat_path."
                    )
                dyn_stats = load_dynamics_stats(dyn_stat_path)
                self.dynamics.set_normalization_stats(
                    action_min=dyn_stats["dyn_action_01"],
                    action_max=dyn_stats["dyn_action_99"],
                    pose_min=dyn_stats["dyn_pose_01"],
                    pose_max=dyn_stats["dyn_pose_99"],
                )
                logging.info("Loaded dynamics normalization stats from %s", dyn_stat_path)

            self.dynamics.eval()
            logging.info(
                "Loaded dynamics model from %s (action_num=%d)",
                dyn_ckpt_path,
                int(self.dynamics.action_num),
            )

        with open(args.data_stat_path, "r") as f:
            data_stat = json.load(f)

        if "wm_state_01" not in data_stat or "wm_state_99" not in data_stat:
            raise ValueError(
                "Normalization file must contain canonical keys wm_state_01/wm_state_99: "
                f"{args.data_stat_path}"
            )
        self.action_p01 = np.asarray(data_stat["wm_state_01"], dtype=np.float32)[None, :]
        self.action_p99 = np.asarray(data_stat["wm_state_99"], dtype=np.float32)[None, :]
        logging.info("Loaded WM normalization bounds (wm_state_01/wm_state_99) from %s", args.data_stat_path)

        if self.action_p01.shape[-1] != args.action_dim or self.action_p99.shape[-1] != args.action_dim:
            raise ValueError(
                f"Normalization dim mismatch: bounds are "
                f"{self.action_p01.shape[-1]}/{self.action_p99.shape[-1]}, "
                f"but action_dim={args.action_dim}."
            )

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def encode_views(self, view_images: list[np.ndarray]) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Encode 3 views to CtrlWorld latent format."""
        vae = self.model.pipeline.vae
        per_view_latents: list[torch.Tensor] = []
        frame_latents: list[torch.Tensor] = []

        with torch.no_grad():
            for image in view_images:
                image_t = torch.from_numpy(np.ascontiguousarray(image)).to(self.dtype).to(self.device)
                x = image_t.permute(2, 0, 1).unsqueeze(0) / 255.0 * 2 - 1.0
                latent = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
                per_view_latents.append(latent)
                frame_latents.append(latent[0])

        merged_latent = torch.cat(frame_latents, dim=1).unsqueeze(0)
        assert merged_latent.shape == (1, 4, 72, 40), f"Expected (1,4,72,40), got {merged_latent.shape}"
        return per_view_latents, merged_latent

    def forward_wm(self, action_cond, current_latent, his_cond=None, text=None):
        args = self.args
        image_cond = current_latent

        # action_cond is 7D WM conditioning.
        # - First 6 dims (continuous pose) are normalized by wm_state_*.
        # - Last dim (gripper) is passed through without percentile normalization.
        pose6 = np.asarray(action_cond[:, :6], dtype=np.float32)
        gripper = np.asarray(action_cond[:, 6:7], dtype=np.float32)
        p01_6 = np.asarray(self.action_p01[:, :6], dtype=np.float32)
        p99_6 = np.asarray(self.action_p99[:, :6], dtype=np.float32)

        pose6_norm = self.normalize_bound(pose6, p01_6, p99_6, clip_min=-1, clip_max=1)
        action_cond = np.concatenate([pose6_norm, gripper], axis=-1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames + args.num_history, args.action_dim)

        with torch.no_grad():
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)

            pipeline = self.model.pipeline
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
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
                frame_level_cond=True,
                his_cond_zero=args.his_cond_zero,
            )

        # (1, F, 4, 72, 40) -> (3, F, 4, 24, 40)
        latents = einops.rearrange(latents, "b f c (m h) (n w) -> (b m n) f c h w", m=3, n=1)

        # decode predicted video
        decoded_video = []
        bsz, frame_num = latents.shape[:2]
        x = latents.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[i : i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        # (F, H, 3W, C) for visualization
        videos_cat = np.concatenate([videos[v] for v in range(videos.shape[0])], axis=2)

        video_dict_pred = [videos[v] for v in range(videos.shape[0])]
        predict_latents = [latents[v] for v in range(latents.shape[0])]
        return videos_cat, video_dict_pred, predict_latents

    @staticmethod
    def _prepare_policy_image(image: np.ndarray, image_size: int = 224) -> np.ndarray:
        # WM outputs 192x320; adapt directly to policy-required 224x224 via interpolation.
        # This avoids padding black borders and avoids an extra resampling stage.
        return _resize_interpolate(image, image_size, image_size)

    def _apply_gripper_update(self, action_gripper: float) -> float:
        # pi05_star_libero(_infer) semantics:
        #   - translation/rotation channels are deltas
        #   - gripper channel is absolute
        return float(action_gripper)

    def _rollout_action_chunk(self, pose7: np.ndarray, state8: np.ndarray, action_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Roll out policy action chunk.

        Returns:
            pose_seq: future absolute EEF Cartesian poses (7D), one per input action.
            state_seq: future policy state sequence (8D), one per input action.
        """
        # Current pose/state are provided separately to the world model. This function
        # keeps a one-to-one mapping:
        #   N delta-EEF actions -> N future absolute poses/states.
        pose_seq = []
        state_seq = []

        if self.dynamics is not None:
            with torch.no_grad():
                horizon = len(action_chunk)
                if horizon > 0:
                    act6 = np.asarray(action_chunk[:horizon, :6], dtype=np.float32)
                    cur_pose6 = np.asarray(pose7[:6], dtype=np.float32)
                    cur_pose6_t = torch.from_numpy(cur_pose6).to(self.device).unsqueeze(0)
                    act6_t = torch.from_numpy(act6).to(self.device).unsqueeze(0)
                    pred_pose6 = (
                        self.dynamics.predict(cur_pose6_t, act6_t, strict_horizon=False)[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    for step_i in range(horizon):
                        next_pose = pose7.copy()
                        next_pose[:6] = pred_pose6[step_i]
                        next_pose[6] = self._apply_gripper_update(float(action_chunk[step_i][6]))

                        next_state = state8.copy()
                        next_state[:7] = next_pose
                        next_state[7] = next_pose[6]

                        pose_seq.append(next_pose)
                        state_seq.append(next_state)

            return np.asarray(pose_seq, dtype=np.float32), np.asarray(state_seq, dtype=np.float32)

        # Build absolute cartesian pose trajectory for WM `action_cond`.
        # Rotation channels are interpreted as axis-angle deltas and composed in
        # quaternion space for numerical stability; exported back to axis-angle.
        # Gripper is treated as absolute at every step.
        cur_pose = pose7.copy()
        cur_state = state8.copy()
        for step_i, act in enumerate(action_chunk):
            next_pose = cur_pose.copy()
            next_state = cur_state.copy()

            # Delta xyz -> absolute xyz.
            next_pose[:3] = next_pose[:3] + act[:3]

            # Delta axis-angle -> quaternion composition -> axis-angle.
            delta_rot = np.asarray(act[3:6], dtype=np.float64)
            delta_norm = float(np.linalg.norm(delta_rot))
            if self.args.rot_delta_norm_clip is not None and delta_norm > float(self.args.rot_delta_norm_clip):
                delta_rot = delta_rot / (delta_norm + 1e-12) * float(self.args.rot_delta_norm_clip)
                delta_norm = float(np.linalg.norm(delta_rot))

            if delta_norm > float(self.args.rot_delta_norm_fail):
                logging.error(
                    "Large delta-rotation norm at step=%d: %.4f rad (> fail %.4f).",
                    step_i,
                    delta_norm,
                    float(self.args.rot_delta_norm_fail),
                )
            elif delta_norm > float(self.args.rot_delta_norm_warn):
                logging.warning(
                    "High delta-rotation norm at step=%d: %.4f rad (> warn %.4f).",
                    step_i,
                    delta_norm,
                    float(self.args.rot_delta_norm_warn),
                )

            q_cur = _rotvec_to_quat(cur_pose[3:6])
            q_delta = _rotvec_to_quat(delta_rot)
            if bool(self.args.quat_delta_compose_local):
                q_next = _quat_mul(q_cur, q_delta)
            else:
                q_next = _quat_mul(q_delta, q_cur)
            q_next = _quat_normalize(q_next)
            next_pose[3:6] = _quat_to_rotvec(q_next).astype(np.float32)

            # Absolute gripper command.
            next_pose[6] = self._apply_gripper_update(act[6])

            # Policy state semantics for LIBERO: 6D absoulute eef positions (3D position + 3D axis-angle) + 2D gripper.
            # We do not run IK in this script, so keep joints unchanged and only update gripper.
            next_state[:7] = next_pose
            next_state[7] = next_pose[6]

            pose_seq.append(next_pose)
            state_seq.append(next_state)
            cur_pose = next_pose
            cur_state = next_state

        return np.asarray(pose_seq, dtype=np.float32), np.asarray(state_seq, dtype=np.float32)

    def forward_policy(self, videos, state8, pose7, text):
        base_idx = self.args.policy_base_camera_idx
        wrist_idx = self.args.policy_wrist_camera_idx

        # WM predicts 192x320 frames; adapt to policy-required 224x224 here.
        image = self._prepare_policy_image(videos[base_idx], image_size=224)
        wrist_image = self._prepare_policy_image(videos[wrist_idx], image_size=224)

        example = {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": state8.astype(np.float32),
            "prompt": str(text),
            "adv_ind": self.args.adv_ind_input,
        }
        action_chunk = np.asarray(self.policy.infer(example)["actions"], dtype=np.float32)

        if action_chunk.ndim != 2 or action_chunk.shape[1] < 7:
            raise ValueError(f"Unexpected action chunk shape: {action_chunk.shape}")

        if action_chunk.shape[0] < self.args.action_horizon:
            pad_n = self.args.action_horizon - action_chunk.shape[0]
            action_chunk = np.concatenate([action_chunk, np.repeat(action_chunk[-1:], pad_n, axis=0)], axis=0)
        action_chunk = action_chunk[: self.args.action_horizon, :7]

        pose_chunk, state_chunk = self._rollout_action_chunk(pose7, state8, action_chunk)

        # raw-time -> 5Hz WM mapping.
        # Build raw-time full sequence by prepending current state:
        #   [t, t+1, ..., t+15]
        # then take fixed 5Hz indices [0, 3, 6, 9, 12] for WM [t..t+4].
        wm_stride = int(getattr(self.args, "policy_downsample_stride", 3))
        if wm_stride != 3:
            raise ValueError(f"policy_downsample_stride must be 3, got {wm_stride}")

        wm_idx = np.asarray([i * wm_stride for i in range(int(self.args.pred_step))], dtype=np.int64)
        action_idx = np.asarray([i * wm_stride for i in range(max(0, int(self.args.pred_step) - 1))], dtype=np.int64)

        pose_seq_full = np.concatenate([pose7[None, :], pose_chunk], axis=0)
        state_seq_full = np.concatenate([state8[None, :], state_chunk], axis=0)

        if pose_seq_full.shape[0] <= int(wm_idx[-1]) or state_seq_full.shape[0] <= int(wm_idx[-1]):
            raise ValueError(
                "Insufficient raw-time rollout for WM 5Hz mapping. "
                f"need max wm idx {int(wm_idx[-1])}, got pose_seq_full={pose_seq_full.shape[0]}, "
                f"state_seq_full={state_seq_full.shape[0]}"
            )
        if action_chunk.shape[0] <= int(action_idx[-1]) if action_idx.size > 0 else False:
            raise ValueError(
                "Insufficient raw action horizon for 5Hz transition mapping. "
                f"need max action idx {int(action_idx[-1])}, got {action_chunk.shape[0]}"
            )

        pose_ds = pose_seq_full[wm_idx].astype(np.float32)
        state_ds = state_seq_full[wm_idx].astype(np.float32)

        # Save rollout transitions on 5Hz timeline:
        # action indices [0, 3, 6, 9] align with
        # (obs_t->obs_t+1), (obs_t+1->obs_t+2), (obs_t+2->obs_t+3), (obs_t+3->obs_t+4).
        actions_sparse_transition = action_chunk[action_idx].astype(np.float32)

        assert pose_ds.shape[0] == self.args.pred_step, (
            f"pose_ds first dim must be pred_step={self.args.pred_step}, got {pose_ds.shape}"
        )
        assert state_ds.shape[0] == self.args.pred_step, (
            f"state_ds first dim must be pred_step={self.args.pred_step}, got {state_ds.shape}"
        )
        assert actions_sparse_transition.shape[0] == max(0, self.args.pred_step - 1), (
            f"actions_sparse_transition first dim must be pred_step-1={self.args.pred_step - 1}, "
            f"got {actions_sparse_transition.shape}"
        )
        assert actions_sparse_transition.shape[1] == 7, (
            f"actions_sparse_transition second dim must be 7, got {actions_sparse_transition.shape}"
        )

        policy_in_out = {
            "actions_raw": action_chunk,
            "pose_chunk_raw": pose_chunk,
            "state_chunk_raw": state_chunk,
            "pose_downsampled": pose_ds,
            "state_downsampled": state_ds,
            # Directly selected policy delta actions aligned with sparse observations.
            "actions_sparse_transition": actions_sparse_transition,
            "wm_pose_indices": wm_idx,
            "wm_action_indices": action_idx,
        }

        return policy_in_out, actions_sparse_transition, pose_ds, state_ds


def load_init_gt_manifest(path: str) -> dict:
    manifest_path = pathlib.Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Initial GT manifest not found: {manifest_path}")

    with open(manifest_path, "r") as f:
        payload = json.load(f)

    entries = payload.get("entries", [])
    if not entries:
        raise ValueError(f"No entries found in initial GT manifest: {manifest_path}")

    for idx, entry in enumerate(entries):
        npz_rel = entry.get("npz")
        if npz_rel is None:
            raise ValueError(f"Entry {idx} missing 'npz' field")
        npz_path = (manifest_path.parent / npz_rel).resolve()
        entry["npz_path"] = str(npz_path)
    payload["entries"] = entries
    return payload


def load_init_episode(entry: dict):
    npz_path = pathlib.Path(entry["npz_path"])
    if not npz_path.exists():
        raise FileNotFoundError(f"Initial GT npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    base_img = np.asarray(data["base_img"], dtype=np.uint8)
    wrist_img = np.asarray(data["wrist_img"], dtype=np.uint8)
    state8 = np.asarray(data["state8"], dtype=np.float32).reshape(-1)
    pose7 = np.asarray(data["pose7"], dtype=np.float32).reshape(-1)
    task_description = str(data["task_description"].item()) if np.asarray(data["task_description"]).ndim == 0 else str(data["task_description"])

    if state8.shape[0] != 8:
        raise ValueError(f"state8 must be 8D, got {state8.shape}")
    if pose7.shape[0] != 7:
        raise ValueError(f"pose7 must be 7D, got {pose7.shape}")

    return base_img, wrist_img, state8, pose7, task_description


def preprocess_for_writer(image: np.ndarray, image_size: int = 256) -> np.ndarray:
    return _resize_interpolate(image, image_size, image_size)


def preprocess_for_wm(image: np.ndarray, height: int, width: int) -> np.ndarray:
    # Initial GT is 256x256; WM expects 192x320.
    # Keep float interpolation output to match training latent extraction behavior
    # (avoid an extra uint8 round-trip before VAE encoding).
    return _resize_interpolate_float(image, height, width)


def build_history_indices_from_buffer(args, history_len: int) -> list[int]:
    """Build history indices on rollout 5Hz buffer.

    Buffer semantics: adjacent entries are adjacent 5Hz timesteps.
    Caller keeps buffer length <= wm_history_cache_len, so index 0 is the
    oldest available frame in current cache window.
    For current index `c = history_len - 1`, return:
      [oldest, c-12, c-9, c-6, c-4, c-2, c-1]
    with clipping to oldest (index 0).
    """
    if history_len <= 0:
        raise ValueError("history_len must be positive")

    current = history_len - 1
    oldest = 0
    rel = list(getattr(args, "history_relative_offsets", [-12, -9, -6, -4, -2, -1]))
    if len(rel) != int(args.num_history) - 1:
        raise ValueError(
            f"history_relative_offsets length must be num_history-1={args.num_history - 1}, got {len(rel)}"
        )

    idx = [oldest] + [max(oldest, current + int(o)) for o in rel]
    if len(idx) != int(args.num_history):
        raise ValueError(f"Expected {args.num_history} history indices, got {len(idx)}")
    return idx


def merge_args(cfg, cli_args):
    for k, v in vars(cli_args).items():
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--adv_ind_input", type=str, default=None)
    parser.add_argument("--init_gt_manifest", type=str, default=None)

    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--pi_ckpt", type=str, default=None)
    parser.add_argument("--data_stat_path", type=str, default=None)

    parser.add_argument("--task_suite_name", type=str, default=None)
    parser.add_argument("--task_ids", type=int, nargs="*", default=None)
    parser.add_argument("--target_rollouts_per_task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--pred_step", type=int, default=None)
    parser.add_argument("--interact_num", type=int, default=None)
    parser.add_argument("--policy_downsample_stride", type=int, default=None)

    parser.add_argument("--policy_base_camera_idx", type=int, default=None)
    parser.add_argument("--policy_wrist_camera_idx", type=int, default=None)
    parser.add_argument("--gripper_max", type=float, default=None)

    parser.add_argument("--save_video", type=str2bool, default=None)
    parser.add_argument("--save_info", type=str2bool, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    parser.add_argument("--save_lerobot_rollout", type=str2bool, default=None)
    parser.add_argument("--rollout_repo_id", type=str, default=None)
    parser.add_argument("--rollout_output_dir", type=str, default=None)
    parser.add_argument("--rollout_overwrite", type=str2bool, default=None)
    parser.add_argument("--rollout_robot_type", type=str, default=None)
    parser.add_argument("--rollout_fps", type=int, default=None)
    parser.add_argument("--rollout_penalty_value", type=float, default=None)

    # Optional learned pose dynamics for policy rollout
    parser.add_argument("--use_dynamics", type=str2bool, default=None)
    parser.add_argument("--dyn_ckpt_path", type=str, default=None)
    parser.add_argument("--dyn_stat_path", type=str, default=None)
    parser.add_argument("--dyn_action_num", type=int, default=None)
    parser.add_argument("--dyn_hidden_size", type=int, default=None)
    parser.add_argument("--dyn_num_layers", type=int, default=None)

    return parser


def main():
    parser = build_parser()
    args_new = parser.parse_args()

    args = wm_args()
    args = merge_args(args, args_new)

    # validate post-merge dataclass constraints
    args.__post_init__()

    export_args = RolloutExportArgs(
        save_lerobot_rollout=getattr(args, "save_lerobot_rollout", True),
        rollout_repo_id=getattr(args, "rollout_repo_id", "ybpy/libero_wm_rollout"),
        rollout_output_dir=getattr(args, "rollout_output_dir", None),
        rollout_overwrite=getattr(args, "rollout_overwrite", False),
        rollout_robot_type=getattr(args, "rollout_robot_type", "panda"),
        rollout_fps=getattr(args, "rollout_fps", 10),
        rollout_penalty_value=getattr(args, "rollout_penalty_value", -1.0),
    )

    writer = None
    if export_args.save_lerobot_rollout:
        writer = LiberoRolloutLeRobotWriter(export_args, image_size=256)

    np.random.seed(args.seed)
    agent = Agent(args)
    print(f"[INFO] history frames used by world model: {args.num_history}")
    logging.info(
        "Using fixed action semantics: delta xyz + delta axis-angle (quaternion composition) + absolute gripper."
    )

    init_gt_manifest = getattr(args, "init_gt_manifest", None)
    if not init_gt_manifest:
        raise ValueError("--init_gt_manifest is required. Generate it with examples/libero/get_init_gt.py first.")

    manifest = load_init_gt_manifest(init_gt_manifest)
    all_entries = manifest["entries"]

    selected_entries = [e for e in all_entries if e.get("task_suite_name", args.task_suite_name) == args.task_suite_name]
    if not selected_entries:
        available_suites = sorted(list({str(e.get("task_suite_name", "<missing>")) for e in all_entries}))
        raise ValueError(
            f"No entries for task_suite_name={args.task_suite_name} in manifest. "
            f"Available suites: {available_suites}"
        )

    if args.task_ids is not None:
        task_id_set = set(args.task_ids)
        selected_entries = [e for e in selected_entries if int(e.get("task_id", -1)) in task_id_set]
        if not selected_entries:
            raise ValueError("No entries matched both --task_suite_name and --task_ids in init GT manifest")

    interact_num = args.interact_num
    pred_step = args.pred_step

    entries_by_task: dict[int, list[dict]] = {}
    for entry in selected_entries:
        task_id = int(entry.get("task_id", -1))
        entries_by_task.setdefault(task_id, []).append(entry)

    for task_id in entries_by_task:
        entries_by_task[task_id].sort(key=lambda x: int(x.get("init_state_idx", x.get("episode_idx", 0))))

    target_rollouts_per_task = int(getattr(args, "target_rollouts_per_task", 1))
    if target_rollouts_per_task <= 0:
        raise ValueError("target_rollouts_per_task must be positive")

    for task_id in sorted(entries_by_task.keys()):
        task_entries = entries_by_task[task_id]
        for rollout_idx in range(target_rollouts_per_task):
            entry = task_entries[rollout_idx % len(task_entries)]
            source_init_idx = int(entry.get("init_state_idx", entry.get("episode_idx", 0)))

            base_img, wrist_img, init_state8, init_pose7, task_description = load_init_episode(entry)

            # Initial GT is stored as flipped raw LIBERO images (typically 256x256).
            # Adapt to WM-required per-view size (default 192x320) before VAE encoding.
            base_img_wm = preprocess_for_wm(base_img, height=args.height, width=args.width)
            wrist_img_wm = preprocess_for_wm(wrist_img, height=args.height, width=args.width)

            zero_img = np.zeros_like(base_img_wm)
            _, first_latent = agent.encode_views([base_img_wm, wrist_img_wm, zero_img])

            # Warm start on the same 5Hz buffer semantics.
            # Early rollout naturally clips long-range history to oldest.
            warm_repeats = int(getattr(args, "history_init_repeats", 4))
            max_cache = int(getattr(args, "wm_history_cache_len", 56))
            init_len = max(args.num_history * warm_repeats, max_cache)
            his_cond = [first_latent for _ in range(init_len)][-max_cache:]
            his_state = [init_state8[None, :] for _ in range(init_len)][-max_cache:]
            his_eef = [init_pose7[None, :] for _ in range(init_len)][-max_cache:]

            # 3 views, each (1, H, W, C)
            video_dict_pred = [
                np.expand_dims(base_img_wm, axis=0),
                np.expand_dims(wrist_img_wm, axis=0),
                np.expand_dims(zero_img, axis=0),
            ]

            video_to_save, info_to_save = [], []
            rollout_steps = []

            for interact_i in range(interact_num):
                current_obs = [v[-1] for v in video_dict_pred]
                current_state = his_state[-1][0]
                current_pose = his_eef[-1][0]

                print("################ policy forward ####################")
                policy_in_out, action_sparse, pose_ds, state_ds = agent.forward_policy(
                    videos=current_obs,
                    state8=current_state,
                    pose7=current_pose,
                    text=task_description,
                )

                print("################ world model forward ################")
                print(
                    f"task: {task_description}, task_id: {task_id}, "
                    f"rollout: {rollout_idx + 1}/{target_rollouts_per_task}, "
                    f"source_init: {source_init_idx}, "
                    f"interact: {interact_i + 1}/{interact_num}"
                )

                history_indices = build_history_indices_from_buffer(args, len(his_eef))
                history_pose = np.concatenate([his_eef[idx] for idx in history_indices], axis=0)
                action_cond = np.concatenate([history_pose, pose_ds], axis=0)
                his_latent = torch.cat([his_cond[idx] for idx in history_indices], dim=0).unsqueeze(0)
                current_latent = his_cond[-1]

                videos_cat, video_dict_pred, predict_latents = agent.forward_wm(
                    action_cond=action_cond,
                    current_latent=current_latent,
                    his_cond=his_latent,
                    text=task_description if args.text_cond else None,
                )

                print("################ record information ################")
                # Save sparse transitions strictly aligned as:
                #   (obs_j, action_j) -> obs_{j+1}
                # Therefore only first (pred_step-1) observations have valid outgoing actions.
                for step_j in range(max(0, pred_step - 1)):
                    base_out = preprocess_for_writer(video_dict_pred[args.policy_base_camera_idx][step_j], image_size=256)
                    wrist_out = preprocess_for_writer(video_dict_pred[args.policy_wrist_camera_idx][step_j], image_size=256)

                    state_to_save = np.asarray(state_ds[step_j], dtype=np.float32)
                    if state_to_save.shape[0] != 8:
                        if state_to_save.shape[0] > 8:
                            state_to_save = state_to_save[:8]
                        else:
                            state_to_save = np.pad(state_to_save, (0, 8 - state_to_save.shape[0]))

                    action_to_save = np.asarray(action_sparse[step_j], dtype=np.float32)

                    rollout_steps.append(
                        {
                            "image": np.ascontiguousarray(base_out),
                            "wrist_image": np.ascontiguousarray(wrist_out),
                            "state": state_to_save,
                            "actions": action_to_save,
                        }
                    )

                video_to_save.append(videos_cat)
                info_to_save.append(policy_in_out)

                # Keep history buffer as contiguous 5Hz timeline:
                # current t is already in buffer; append only future [t+1..t+4].
                for step_j in range(1, pred_step):
                    his_state.append(np.asarray(state_ds[step_j], dtype=np.float32)[None, :])
                    his_eef.append(np.asarray(pose_ds[step_j], dtype=np.float32)[None, :])
                    his_cond.append(torch.cat([v[step_j] for v in predict_latents], dim=1).unsqueeze(0))

                # cap history cache length (shared with training oldest definition)
                max_cache = int(getattr(args, "wm_history_cache_len", 56))
                if len(his_state) > max_cache:
                    his_state = his_state[-max_cache:]
                    his_eef = his_eef[-max_cache:]
                    his_cond = his_cond[-max_cache:]

            print("##########################################################################")
            episode_tag = f"task{task_id:03d}_roll{rollout_idx:03d}_src{source_init_idx:03d}"
            text_id = task_description.replace(" ", "_").replace(",", "").replace(".", "").replace("'", "").replace('"', "")[:40]
            uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            if args.save_video and len(video_to_save) > 0:
                video = np.concatenate(video_to_save, axis=0)
                filename_video = (
                    f"{args.save_dir}/{args.task_name}/video/"
                    f"{args.config_name}_time_{uuid}_{episode_tag}_{text_id}.mp4"
                )
                pathlib.Path(filename_video).parent.mkdir(parents=True, exist_ok=True)
                mediapy.write_video(filename_video, video, fps=args.rollout_fps)
                print(f"Saving video to {filename_video}")

            # TODO: replace with a learned rollout-success classifier (final frame or full video).
            rollout_success = True
            if args.save_info and len(info_to_save) > 0:
                info = {
                    "success": int(rollout_success),
                    "task_suite_name": str(args.task_suite_name),
                    "task_id": int(task_id),
                    "rollout_idx": int(rollout_idx),
                    "source_init_state_idx": int(source_init_idx),
                    "instructions": task_description,
                    "interact_num": int(interact_num),
                    "pred_step": int(pred_step),
                }
                for key in info_to_save[0].keys():
                    info[key] = []
                    for k in range(len(info_to_save)):
                        if isinstance(info_to_save[k][key], np.ndarray):
                            info[key] += info_to_save[k][key].tolist()

                filename_info = (
                    f"{args.save_dir}/{args.task_name}/info/"
                    f"{args.config_name}_time_{uuid}_{episode_tag}_{text_id}.json"
                )
                pathlib.Path(filename_info).parent.mkdir(parents=True, exist_ok=True)
                with open(filename_info, "w") as f:
                    json.dump(info, f, indent=4)
                print(f"Saving trajectory info to {filename_info}")

            if writer is not None:
                writer.save_episode(steps=rollout_steps, task=task_description, success=rollout_success)
                print(f"Saved LeRobot episode to {writer.dataset_path}")

            print("##########################################################################")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
