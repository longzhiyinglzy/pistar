"""Extract SVD latents and minimal canonical LIBERO annotations.

Input (LeRobot parquet episodes):
  <lerobot_root>/meta/episodes.jsonl
  <lerobot_root>/data/chunk-xxx/episode_xxxxxx.parquet

Output (WM-ready dataset):
  <output_path>/videos/{train,val}/{episode_id}/{0,1,2}.mp4
  <output_path>/latent_videos/{train,val}/{episode_id}/{0,1,2}.pt
  <output_path>/annotation/{train,val}/{episode_id}.json

Semantic contract (canonical LeRobot fields only):
    - actions: delta EEF action [7] = delta pose6 + gripper command1
    - abs_pose: WM/Dynamics absolute pose [7] = abs pose6 + gripper command1

Task text source contract (aligned with LeRobot canonical metadata layout):
    - `task` is not a parquet feature in this dataset.
    - Task text is resolved from LeRobot meta files.
    - Preferred source: meta/episodes.jsonl -> episode-level `tasks`.
    - Fallback source: meta/tasks.jsonl + parquet `task_index`.

Output annotation keeps only minimal fields required by downstream WM/Dynamics.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import mediapy
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers.models import AutoencoderKLTemporalDecoder
from PIL import Image
from tqdm import tqdm


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value}")


def _episode_output_paths(output_root: Path, split: str, episode_id: int) -> dict[str, Path]:
    ep_dir = str(int(episode_id))
    return {
        "video0": output_root / "videos" / split / ep_dir / "0.mp4",
        "video1": output_root / "videos" / split / ep_dir / "1.mp4",
        "video2": output_root / "videos" / split / ep_dir / "2.mp4",
        "latent0": output_root / "latent_videos" / split / ep_dir / "0.pt",
        "latent1": output_root / "latent_videos" / split / ep_dir / "1.pt",
        "latent2": output_root / "latent_videos" / split / ep_dir / "2.pt",
        "annotation": output_root / "annotation" / split / f"{int(episode_id)}.json",
    }


def _episode_outputs_exist(output_root: Path, split: str, episode_id: int) -> bool:
    paths = _episode_output_paths(output_root, split, episode_id)
    return all(p.exists() for p in paths.values())


def _prepare_output_root(output_path: str, overwrite: bool) -> Path:
    out = Path(output_path).resolve()
    if overwrite and out.exists():
        # Guard against accidental dangerous deletion.
        if str(out) in {"/", out.anchor}:
            raise ValueError(f"Refusing to remove unsafe output path: {out}")
        print(f"[INFO] overwrite=True, removing existing output directory: {out}")
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "latent_videos").mkdir(parents=True, exist_ok=True)
    (out / "annotation").mkdir(parents=True, exist_ok=True)
    return out


def _decode_image_cell(cell: Any, root_dir: Path) -> np.ndarray:
    if isinstance(cell, np.ndarray):
        img = cell
    elif isinstance(cell, Image.Image):
        img = np.asarray(cell)
    elif isinstance(cell, (bytes, bytearray)):
        img = iio.imread(io.BytesIO(cell))
    elif isinstance(cell, dict):
        if cell.get("bytes") is not None:
            img = iio.imread(io.BytesIO(cell["bytes"]))
        elif cell.get("path"):
            p = Path(cell["path"])
            if not p.is_absolute():
                p = root_dir / p
            img = iio.imread(p)
        else:
            raise ValueError(f"Unsupported image dict keys: {list(cell.keys())}")
    else:
        raise ValueError(f"Unsupported image cell type: {type(cell)}")

    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    return img.astype(np.uint8)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class EncodeLatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        old_path: str,
        new_path: str,
        svd_path: str,
        device: torch.device,
        *,
        size: tuple[int, int] = (192, 320),
        rgb_skip: int = 3,
        skip_existing: bool = True,
    ):
        self.old_path = Path(old_path)
        self.new_path = Path(new_path)
        self.size = size
        self.skip = int(rgb_skip)
        self.skip_existing = bool(skip_existing)

        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_path, subfolder="vae").to(device)
        self.episodes = _load_jsonl(self.old_path / "meta" / "episodes.jsonl")
        self.episode_meta_by_id = {
            int(ep["episode_index"]): ep for ep in self.episodes if "episode_index" in ep
        }

        self.task_text_by_index: dict[int, str] = {}
        tasks_path = self.old_path / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            for row in _load_jsonl(tasks_path):
                if "task_index" not in row:
                    continue
                key = int(row["task_index"])
                value = str(row.get("task", "")).strip()
                if value and value.lower() != "nan":
                    self.task_text_by_index[key] = value

    def __len__(self):
        return len(self.episodes)

    def _encode_video_latent(self, frames_uint8: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        frames = torch.tensor(frames_uint8).permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
        frames = frames[:: self.skip]
        x = torch.nn.functional.interpolate(frames, size=self.size, mode="bilinear", align_corners=False)
        x = x.to(self.vae.device)

        with torch.no_grad():
            latents = []
            batch_size = 64
            for i in range(0, len(x), batch_size):
                batch = x[i : i + batch_size]
                lat = self.vae.encode(batch).latent_dist.sample().mul_(self.vae.config.scaling_factor).cpu()
                latents.append(lat)
            lat = torch.cat(latents, dim=0)

        resize_video = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
        resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        return lat, resize_video

    def _resolve_instruction(self, episode_id: int, parquet_path: Path) -> str:
        """Resolve task text for one episode.

        Priority:
        1) meta/episodes.jsonl: episode-level `tasks[0]`
        2) meta/tasks.jsonl + parquet `task_index`
        3) otherwise raise explicit error
        """

        def _valid_text(value: Any) -> bool:
            s = str(value).strip()
            return (len(s) > 0) and (s.lower() != "nan")

        episode_found = episode_id in self.episode_meta_by_id
        episode_tasks_present = False
        episode_task_valid = False

        if episode_found:
            ep_meta = self.episode_meta_by_id[episode_id]
            tasks = ep_meta.get("tasks")
            if isinstance(tasks, list) and len(tasks) > 0:
                episode_tasks_present = True
                candidate = tasks[0]
                if _valid_text(candidate):
                    episode_task_valid = True
                    return str(candidate).strip()

        task_index_present = False
        task_index_value = None
        task_text_found = False
        task_text_valid = False

        # Fallback only when episode-level tasks are unavailable/invalid.
        try:
            task_df = pd.read_parquet(parquet_path, columns=["task_index"])
            if len(task_df) > 0 and "task_index" in task_df.columns:
                raw_task_index = np.asarray(task_df["task_index"].iloc[0]).reshape(-1)
                if raw_task_index.size > 0:
                    task_index_present = True
                    task_index_value = int(raw_task_index[0])
                    candidate = self.task_text_by_index.get(task_index_value)
                    task_text_found = candidate is not None
                    if candidate is not None and _valid_text(candidate):
                        task_text_valid = True
                        return str(candidate).strip()
        except Exception:
            task_index_present = False

        raise ValueError(
            "Failed to resolve instruction from LeRobot metadata. "
            f"episode_id={episode_id}, parquet_path={parquet_path}, "
            f"episode_found_in_episodes_jsonl={episode_found}, "
            f"episode_tasks_present={episode_tasks_present}, "
            f"episode_task_valid={episode_task_valid}, "
            f"task_index_present_in_parquet={task_index_present}, "
            f"task_index={task_index_value}, "
            f"task_found_in_tasks_jsonl={task_text_found}, "
            f"task_text_valid={task_text_valid}. "
            "Both episodes.jsonl and tasks.jsonl fallback failed."
        )

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        episode_id = int(ep.get("episode_index", idx))
        chunk_id = episode_id // 1000
        data_type = "val" if episode_id % 100 == 99 else "train"

        parquet_path = self.old_path / "data" / f"chunk-{chunk_id:03d}" / f"episode_{episode_id:06d}.parquet"
        if not parquet_path.exists():
            return 0

        if self.skip_existing and _episode_outputs_exist(self.new_path, data_type, episode_id):
            print(f"[INFO] skip existing episode {episode_id} ({data_type})")
            return 0

        try:
            # Canonical LeRobot schema only.
            # Task text is NOT a parquet feature for this dataset.
            # Keep parquet schema minimal and resolve text from meta/*.jsonl.
            required_cols = ["image", "wrist_image", "actions", "abs_pose"]
            df = pd.read_parquet(parquet_path, columns=required_cols)

            length = len(df)
            if length <= 0:
                return 0

            base_imgs = []
            wrist_imgs = []
            actions7 = []
            abs_pose7 = []

            for i in range(length):
                base = _decode_image_cell(df["image"].iloc[i], self.old_path)
                wrist = _decode_image_cell(df["wrist_image"].iloc[i], self.old_path)

                a7 = np.asarray(df["actions"].iloc[i], dtype=np.float32).reshape(-1)
                if a7.shape[0] < 7:
                    a7 = np.pad(a7, (0, 7 - a7.shape[0]))
                a7 = a7[:7]

                p7 = np.asarray(df["abs_pose"].iloc[i], dtype=np.float32).reshape(-1)
                if p7.shape[0] < 7:
                    p7 = np.pad(p7, (0, 7 - p7.shape[0]))
                p7 = p7[:7]

                base_imgs.append(base)
                wrist_imgs.append(wrist)
                actions7.append(a7)
                abs_pose7.append(p7)

            instruction = self._resolve_instruction(episode_id=episode_id, parquet_path=parquet_path)
            success = int(bool(ep.get("success", True)))

            base_arr = np.asarray(base_imgs, dtype=np.uint8)
            wrist_arr = np.asarray(wrist_imgs, dtype=np.uint8)
            zero_arr = np.zeros_like(base_arr, dtype=np.uint8)

            (self.new_path / "videos" / data_type / str(episode_id)).mkdir(parents=True, exist_ok=True)
            (self.new_path / "latent_videos" / data_type / str(episode_id)).mkdir(parents=True, exist_ok=True)

            # Third view is a zero placeholder to match 3-view WM latent layout.
            # It is not a raw sensor field in LeRobot source data.
            for view_id, frames_arr in enumerate([base_arr, wrist_arr, zero_arr]):
                lat, resized_video = self._encode_video_latent(frames_arr)
                mediapy.write_video(
                    str(self.new_path / "videos" / data_type / str(episode_id) / f"{view_id}.mp4"),
                    resized_video,
                    fps=5,
                )
                torch.save(lat, self.new_path / "latent_videos" / data_type / str(episode_id) / f"{view_id}.pt")

            actions7 = np.asarray(actions7, dtype=np.float32)
            abs_pose7 = np.asarray(abs_pose7, dtype=np.float32)

            ds_idx = np.arange(0, len(abs_pose7), self.skip, dtype=np.int64)
            abs_pose_ds = abs_pose7[ds_idx]
            if abs_pose_ds.shape[0] <= 0:
                return 0

            info = {
                "texts": [instruction],
                "episode_id": int(episode_id),
                "success": int(success),
                "video_length": int(abs_pose_ds.shape[0]),
                "raw_length": int(abs_pose7.shape[0]),
                "videos": [
                    {"video_path": f"videos/{data_type}/{episode_id}/0.mp4"},
                    {"video_path": f"videos/{data_type}/{episode_id}/1.mp4"},
                    {"video_path": f"videos/{data_type}/{episode_id}/2.mp4"},
                ],
                "latent_videos": [
                    {"latent_video_path": f"latent_videos/{data_type}/{episode_id}/0.pt"},
                    {"latent_video_path": f"latent_videos/{data_type}/{episode_id}/1.pt"},
                    {"latent_video_path": f"latent_videos/{data_type}/{episode_id}/2.pt"},
                ],
                "abs_pose": abs_pose_ds.tolist(),
                "abs_pose_raw": abs_pose7.tolist(),
                "action_raw": actions7.tolist(),
            }

            (self.new_path / "annotation" / data_type).mkdir(parents=True, exist_ok=True)
            with open(self.new_path / "annotation" / data_type / f"{episode_id}.json", "w") as f:
                json.dump(info, f, indent=2)

        except Exception as exc:
            print(f"[WARN] episode {episode_id} failed: {exc}")

        return 0


def main(
    lerobot_root: str = "/public/home/chenyuyao1/.cache/huggingface/lerobot/ybpy/libero_pistar_wm",
    output_path: str = "/public/home/chenyuyao1/code/pistar/dataset/libero_wm",
    svd_path: str = "/public/home/chenyuyao1/model/stable-video-diffusion-img2vid",
    rgb_skip: int = 3,
    debug_max_episodes: int = 0,
    overwrite: bool = False,
    skip_existing: bool = False,
):
    resolved_output = _prepare_output_root(output_path, overwrite=bool(overwrite))
    effective_skip_existing = bool(skip_existing)
    if bool(overwrite) and bool(skip_existing):
        print("[INFO] overwrite=True, skip_existing is ignored.")
        effective_skip_existing = False

    print(f"[INFO] output_path={resolved_output}")
    print(f"[INFO] overwrite={bool(overwrite)}")
    print(f"[INFO] skip_existing={effective_skip_existing}")
    print(f"[INFO] rgb_skip={int(rgb_skip)}")

    accelerator = Accelerator()
    dataset = EncodeLatentDataset(
        old_path=lerobot_root,
        new_path=str(resolved_output),
        svd_path=svd_path,
        device=accelerator.device,
        size=(192, 320),
        rgb_skip=rgb_skip,
        skip_existing=effective_skip_existing,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
    loader = accelerator.prepare_data_loader(loader)

    for idx, _ in enumerate(tqdm(loader, desc="Extracting latents")):
        if debug_max_episodes > 0 and idx >= debug_max_episodes:
            break


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--lerobot_root", type=str, default="/public/home/chenyuyao1/.cache/huggingface/lerobot/ybpy/libero_pistar_wm")
    parser.add_argument("--output_path", type=str, default="/public/home/chenyuyao1/code/pistar/dataset/libero_wm")
    parser.add_argument("--svd_path", type=str, default="/public/home/chenyuyao1/model/stable-video-diffusion-img2vid")
    parser.add_argument("--rgb_skip", type=int, default=3)
    parser.add_argument("--debug_max_episodes", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        help="If true, remove whole output_path before processing.",
    )
    parser.add_argument(
        "--skip_existing",
        type=str2bool,
        default=False,
        help="If true, skip episodes whose video/latent/annotation outputs are all present.",
    )
    args = parser.parse_args()

    main(
        lerobot_root=args.lerobot_root,
        output_path=args.output_path,
        svd_path=args.svd_path,
        rgb_skip=args.rgb_skip,
        debug_max_episodes=args.debug_max_episodes,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
    )
