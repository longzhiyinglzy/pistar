import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_piper_example() -> dict:
    """Creates a random input example for Piper policy."""
    return {
        "state": np.ones((7,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_wrist1": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Inputs for Piper single-arm policy.
    
    Expected inputs:
    - images: dict with "cam_high", "cam_wrist", and optional "cam_wrist1"
      or "observation/images/cam_head", "observation/images/cam_wrist",
      and optional "observation/images/cam_side"
    - state: [7] (6 joints + 1 gripper)
    - actions: [action_horizon, 7]
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_wrist", "cam_wrist1")

    def __call__(self, data: dict) -> dict:
        state_key = "state" if "state" in data else "observation/state"
        state = np.asarray(data[state_key], dtype=np.float32)

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if len(img.shape) == 3 and img.shape[0] in (1, 3):
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        in_images = data["images"]

        def get_image(*keys):
            for key in keys:
                if key in in_images:
                    return in_images[key]
            return None
        
        base_raw = get_image("cam_high", "observation/images/cam_head")
        if base_raw is None:
            raise KeyError(
                "Missing Piper base camera. Expected one of: "
                "cam_high, observation/images/cam_head"
            )

        base_image = convert_image(base_raw)
        
        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        wrist_raw = get_image("cam_wrist", "observation/images/cam_wrist")
        if wrist_raw is not None:
            images["left_wrist_0_rgb"] = convert_image(wrist_raw)
            image_masks["left_wrist_0_rgb"] = np.True_
        else:
            images["left_wrist_0_rgb"] = np.zeros_like(base_image)
            image_masks["left_wrist_0_rgb"] = np.False_

        side_raw = get_image("cam_wrist1", "observation/images/cam_side")
        if side_raw is not None:
            images["right_wrist_0_rgb"] = convert_image(side_raw)
            image_masks["right_wrist_0_rgb"] = np.True_
        else:
            images["right_wrist_0_rgb"] = np.zeros_like(base_image)
            image_masks["right_wrist_0_rgb"] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if "adv_ind" in data:
            inputs["adv_ind"] = data["adv_ind"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """Outputs for Piper policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :7], dtype=np.float32)
        return {"actions": actions}
