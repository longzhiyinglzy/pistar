from pathlib import Path
import json
import os

import cv2
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _normalize_checkpoint_dir(model_path: str) -> str:
    """Normalize common checkpoint path mistakes and validate layout."""
    original_path = Path(model_path)
    candidate_paths = [original_path]

    if "project_wang" in original_path.parts:
        rewritten_parts = [part for part in original_path.parts if part != "project_wang"]
        rewritten_path = Path(*rewritten_parts)
        if rewritten_path != original_path:
            candidate_paths.append(rewritten_path)

    seen = set()
    for candidate in candidate_paths:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)

        path = candidate
        if path.name in {"assets", "params"} and path.parent.exists():
            candidate_root = path.parent
            if (candidate_root / "params").exists() or (candidate_root / "model.safetensors").exists():
                path = candidate_root

        has_jax_layout = (path / "params" / "_METADATA").exists()
        has_pytorch_layout = (path / "model.safetensors").exists()
        if has_jax_layout or has_pytorch_layout:
            if path != original_path:
                print(f"[PI0] Adjust checkpoint dir from {original_path} -> {path}")
            return str(path)

    raise FileNotFoundError(
        "Invalid checkpoint directory: "
        f"{original_path}. Expected either '<ckpt>/params/_METADATA' (JAX) "
        "or '<ckpt>/model.safetensors' (PyTorch)."
    )


def _uses_adv_ind(train_config) -> bool:
    return bool(getattr(getattr(train_config, "model", None), "pistar", False))


def _validate_adv_ind(train_config, adv_ind: str | None) -> tuple[bool, str | None]:
    use_adv_ind = _uses_adv_ind(train_config)
    if use_adv_ind and adv_ind is None:
        raise ValueError(
            f"Config '{train_config.name}' requires adv_ind for PiStar inference. "
            "Pass adv_ind explicitly, for example 'positive' or 'negative'."
        )
    if not use_adv_ind:
        return False, None
    return True, adv_ind


def _load_instruction(task_name: str) -> str:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    possible_paths = [
        os.path.join(root_dir, "task_instructions", f"{task_name}.json"),
        os.path.join(root_dir, "datasets", "instructions", f"{task_name}.json"),
        os.path.join("task_instructions", f"{task_name}.json"),
    ]

    for json_path in possible_paths:
        try:
            with open(json_path, "r") as f_instr:
                instruction_dict = json.load(f_instr)
            instructions = instruction_dict["instructions"]
            instruction = np.random.choice(instructions)
            print(f"successfully set instruction from file: {json_path}")
            print(f"instruction: {instruction}")
            return instruction
        except (FileNotFoundError, OSError, KeyError):
            continue

    print(f"Warning: Instruction file not found in any of: {possible_paths}")
    print(f"Using task_name as instruction: {task_name}")
    return task_name


class PI0_DUAL:
    def __init__(self, model_path, task_name, adv_ind: str | None = None):
        self.task_name = task_name

        train_config_name = "pi0_base_aloha_robotwin_lora"
        config = _config.get_config(train_config_name)
        self.use_adv_ind, self.adv_ind = _validate_adv_ind(config, adv_ind)
        model_path = _normalize_checkpoint_dir(model_path)
        self.policy = _policy_config.create_trained_policy(config, model_path)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.random_set_language()

    def set_img_size(self, img_size):
        self.img_size = img_size

    def random_set_language(self):
        self.instruction = _load_instruction(self.task_name)

    def update_observation_window(self, img_arr, state):
        imgs_array = []

        if isinstance(img_arr[0], bytes):
            for data in img_arr:
                jpeg_bytes = np.array(data).tobytes().rstrip(b"\0")
                nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                imgs_array.append(cv2.imdecode(nparr, 1))
        else:
            imgs_array = img_arr

        img_front, img_right, img_left, _ = imgs_array[0], imgs_array[1], imgs_array[2], state
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }
        if self.use_adv_ind:
            self.observation_window["adv_ind"] = self.adv_ind

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")


class PI0_SINGLE:
    def __init__(self, task_name, train_config_name, model_name, checkpoint_id, adv_ind: str | None = None):
        self.train_config_name = train_config_name
        self.task_name = task_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)
        self.use_adv_ind, self.adv_ind = _validate_adv_ind(config, adv_ind)
        if self.checkpoint_id.startswith("/"):
            model_path = self.checkpoint_id
        else:
            model_path = f"policy/openpi/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}"
        model_path = _normalize_checkpoint_dir(model_path)
        self.policy = _policy_config.create_trained_policy(config, model_path)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.random_set_language()

    def set_img_size(self, img_size):
        self.img_size = img_size

    def random_set_language(self):
        self.instruction = _load_instruction(self.task_name)

    def update_observation_window(self, img_arr, state):
        img_head, img_wrist = img_arr[0], img_arr[1]
        img_side = img_arr[2] if len(img_arr) > 2 and img_arr[2] is not None else np.zeros_like(img_head)
        self.observation_window = {
            "observation/state": state,
            "observation/image": img_head,
            "observation/wrist_image": img_wrist,
            "observation/side_image": img_side,
            "state": state,
            "images": {
                "cam_high": img_head,
                "cam_wrist": img_wrist,
                "cam_wrist1": img_side,
            },
            "prompt": self.instruction,
        }
        if self.use_adv_ind:
            self.observation_window["adv_ind"] = self.adv_ind

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"][:, :8]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")


class PI0_LIBERO:
    def __init__(self, model_path, task_name, train_config_name="pi05_libero_local", adv_ind: str | None = None):
        self.train_config_name = train_config_name
        self.task_name = task_name
        self.model_path = _normalize_checkpoint_dir(model_path)

        config = _config.get_config(self.train_config_name)
        self.use_adv_ind, self.adv_ind = _validate_adv_ind(config, adv_ind)
        self.policy = _policy_config.create_trained_policy(config, self.model_path)
        print(f"loading model success from: {self.model_path}")
        self.img_size = (224, 224)
        self.observation_window = None
        self.random_set_language()

    def set_img_size(self, img_size):
        self.img_size = img_size

    def random_set_language(self):
        self.instruction = _load_instruction(self.task_name)

    def update_observation_window(self, img_arr, state):
        img_high, img_wrist = img_arr[0], img_arr[1]
        img_high = np.transpose(img_high, (2, 0, 1))
        img_wrist = np.transpose(img_wrist, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_high,
                "cam_wrist": img_wrist,
            },
            "prompt": self.instruction,
        }
        if self.use_adv_ind:
            self.observation_window["adv_ind"] = self.adv_ind

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"][:, :7]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language instruction")
