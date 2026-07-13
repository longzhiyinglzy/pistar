# PiStar0.6 Single-Arm Piper SpaceMouse Reproduction

This fork is a real-robot reproduction of [ybpy/pistar](https://github.com/ybpy/pistar) for a single-arm Piper setup driven by SpaceMouse demonstrations and SpaceMouse DAgger intervention. It includes the PiStar/OpenPI code plus the Piper collection, conversion, value-labeling, advantage-labeling, training, serving, and RTC evaluation helpers used in the experiment.

Example task prompt used in the reported run:

```text
Pick up the block1 and assemble it.
```

The current best local SpaceMouse pipeline run used:

```text
base policy: pi0.5 base
demo data: 250 successful LeRobot v2.1 demos
DAgger data: 250 rollout episodes, 200 success + 50 failure
policy data: demo250_positive + success_dagger200_adv repeated 3 times
policy episodes: 850
control rate: 30 Hz
action horizon: 50
best tested guidance beta: 1.2
```

## Repository Notes

This repository tracks code and reproducible workflow documentation. Do not commit robot datasets, checkpoints, VLM weights, Pi0.5 weights, `outputs/`, `.idea/`, or private machine paths.

The public workflow assumes these local paths are set by the user:

```bash
export PISTAR_ROOT=/path/to/pistar
export OPENPI_ROOT=/path/to/openpi
export CONTROL_REPO=$PISTAR_ROOT/control_your_robot
export LEROBOT_ROOT=/path/to/lerobot/piper

export DEMO_V21=$LEROBOT_ROOT/assemble_block1_v21
export DEMO250=$LEROBOT_ROOT/assemble_block1_v21_demo250_uniform
export DAGGER250=$LEROBOT_ROOT/assemble_block1_demo250_dagger_r0
export DAGGER250_ADV=$LEROBOT_ROOT/assemble_block1_demo250_dagger_r0_adv_value_accum4_r0
export DEMO250_PISTAR=$LEROBOT_ROOT/assemble_block1_v21_demo250_uniform_pistar_30hz_3view_positive
export DAGGER_SUCCESS200=$LEROBOT_ROOT/assemble_block1_dagger_success200_adv_r1
export POLICY_DATA=$LEROBOT_ROOT/assemble_block1_policy_demo250_success200_x3_r1

export VLM_ROOT=/path/to/ybpy/vlm_ckpt
export PI05_BASE_PARAMS=/path/to/pi05_base/params

export HEAD_SERIAL=<head_camera_serial>
export SIDE_SERIAL=<side_camera_serial>
export WRIST_SERIAL=<wrist_camera_serial>
export TASK_NAME="Pick up the block1 and assemble it."
```

## Action Representation

The Piper PiStar config uses `extra_delta_transform=True`.

```text
dataset action columns: absolute target joint positions
model training target: delta joint action for joints 1-6
gripper target: absolute
deployment command: absolute target joint position after output transform
```

`DeltaActions(mask=(True, True, True, True, True, True, False))` converts absolute targets into delta targets during training. `AbsoluteActions(...)` converts the predicted delta back to absolute target joints before the Piper deployment script sends the command to the controller. So the model learns relative joint deltas, while the provided Piper runtime still receives absolute target positions.

Verify the active transforms with:

```bash
cd "$PISTAR_ROOT"

PYTHONPATH=src venv/bin/python - <<'PY'
from openpi.training.config import get_config

for name in ["pi05_star_assemble_blocks_h50_from_pi05", "pi05_star_assemble_blocks_h50_from_pi05_infer"]:
    c = get_config(name)
    dc = c.data.create(c.assets_dirs, c.model)
    print(name)
    print("inputs:")
    for x in dc.data_transforms.inputs:
        print(" ", x)
    print("outputs:")
    for x in dc.data_transforms.outputs:
        print(" ", x)
    print("beta:", getattr(c.model, "adv_guidance_beta", None))
    print("discrete_state_input:", c.model.discrete_state_input)
PY
```

Expected summary:

```text
inputs:
  PiperInputs()
  DeltaActions(mask=(True, True, True, True, True, True, False))
outputs:
  AbsoluteActions(mask=(True, True, True, True, True, True, False))
  PiperOutputs()
```

## Dataset Fields

The PiStar flat LeRobot datasets use:

```text
image
side_image
wrist_image
state
actions
intervention
value_label
reward
reward_label
adv_ind
timestamp
frame_index
episode_index
index
task_index
```

The model view mapping is:

```text
image       -> cam_high
wrist_image -> cam_wrist
side_image  -> cam_wrist1
```

## 1. Hardware Setup

Hardware used in the reported run:

```text
robot: Piper single arm
arm name: left_arm
CAN device: can0
state source: joint
cameras: RealSense head + side + wrist
teleop: 3Dconnexion SpaceMouse
policy control rate: 30 Hz
SpaceMouse control rate during DAgger: 200 Hz
action horizon: 50
RTC execution horizon: 10
```

Install and verify robot-side dependencies before running collection:

```bash
python - <<'PY'
import piper_sdk
import pyrealsense2
print("piper_sdk ok")
print("pyrealsense2 ok")
PY
```

Set camera serials either through command-line arguments in the rollout/eval scripts or in `control_your_robot/my_robot/camera_config.py`.

## 2. Collect HDF5 / LeRobot v2.1 Demos With SpaceMouse

The rest of the pipeline expects a successful LeRobot v2.1 demo dataset. You can collect LeRobot data directly or collect HDF5 first and convert it.

Direct LeRobot collection example:

```bash
cd "$PISTAR_ROOT"

python control_your_robot/example/collect/collect_lerobot_spacemouse_piper_teleop.py \
  --repo-id assemble_block1_v21 \
  --output-dir "$LEROBOT_ROOT" \
  --task-name "$TASK_NAME" \
  --num-episode 500 \
  --fps 30 \
  --control-hz 200 \
  --arm-can can0 \
  --max-step 1500 \
  --enter-label success
```

Keyboard controls:

```text
Enter  start or finish according to --enter-label
s      save success
f      save failure
r      discard
q/Esc  quit
```

For the conservative PiStar0.6 experiment, collect 500 successful demos first, then sample 250 uniformly:

```bash
cd "$PISTAR_ROOT"

PYTHONPATH=src venv/bin/python scripts/sample_lerobot_episode_subset.py \
  --source "$DEMO_V21" \
  --output "$DEMO250" \
  --count 250 \
  --strategy stratified \
  --seed 42 \
  --video-mode copy \
  --overwrite
```

Use `--video-mode copy` for a standalone dataset. Training reads parquet/image features, but keeping videos is useful for inspection and strict dataset portability.

## 3. Convert HDF5 to LeRobot v2.1 if Needed

If your raw demonstrations were collected as HDF5 with `control_your_robot`, convert them before continuing:

```bash
cd "$PISTAR_ROOT/control_your_robot"

python scripts/convert2lerobot.py \
  /path/to/hdf5_demo_root \
  piper/assemble_block1_v21
```

After conversion, inspect the dataset:

```bash
python scripts/inspect_parquet.py "$DEMO_V21"
```

The converted dataset should contain 7-D `state` and `actions` columns: six Piper joints plus gripper.

## 4. Train Initial Pi0.5 Policy

Train the initial policy from `pi05_base` on the 250-demo dataset. This step is usually run in an OpenPI training environment.

Compute norm stats in OpenPI using the same dataset and action representation used by your Pi0.5 config. Keep the generated `norm_stats.json` with the checkpoint.

Serve the trained initial policy for rollout:

```bash
cd "$OPENPI_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_Piper_AssembleBlock1_inference \
  --policy.dir /path/to/initial_pi05_demo250_checkpoint \
  --port 8000
```

This initial policy is only used to generate DAgger rollout data. The final PiStar policy below is initialized from `pi05_base`, not from this task-finetuned initial policy.

## 5. Rollout Initial Policy With SpaceMouse DAgger

Collect 250 rollout episodes with sticky SpaceMouse intervention:

```text
target success: 200 episodes
target failure: 50 episodes
sticky intervention: after the first intervention, all later frames are marked intervention
max frames: 1500
```

Run:

```bash
cd "$PISTAR_ROOT"

scripts/run_pi05_rtc_rollout_collect.sh \
  --control-repo-path "$CONTROL_REPO" \
  --server-host localhost \
  --server-port 8000 \
  --repo-id "$(basename "$DAGGER250")" \
  --output-dir "$(dirname "$DAGGER250")" \
  --task-name "$TASK_NAME" \
  --arm-can can0 \
  --arm-name left_arm \
  --state-source joint \
  --cam-head-serial "$HEAD_SERIAL" \
  --cam-side-serial "$SIDE_SERIAL" \
  --cam-wrist-serial "$WRIST_SERIAL" \
  --fps 30 \
  --control-dt 0.033333 \
  --action-horizon 50 \
  --num-episode 250 \
  --max-step 1500 \
  --save-adv-ind none \
  --enter-label discard \
  --min-save-frames 30 \
  --post-reset-sleep 2.0 \
  --enable-spacemouse-intervention \
  --sticky-intervention true \
  --spacemouse-control-hz 200 \
  --rtc-enabled true \
  --rtc-execution-horizon 10 \
  --rtc-max-guidance-weight 10.0 \
  --rtc-prefix-attention-schedule exp \
  --rtc-measure-inference-delay false \
  --rtc-inference-delay-steps 4 \
  --rtc-prefetch-threshold 20 \
  --rtc-worker-sleep 0.005 \
  --rtc-debug false
```

Useful keys:

```text
Enter  start episode
s      save success
f      save failure
r      discard
h      home and mark following frames as intervention
q/Esc  quit
```

Reported collection:

```text
saved_total=250
success=200
failure=50
dagger=64
dagger_rate=25.6%
intervention_frames=18733
discarded=37
```

## 6. Train VLM Value Function

Train value on the full DAgger set, including both success and failure episodes.

```bash
cd "$PISTAR_ROOT"

export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_dagger250_value
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHONPATH=src venv/bin/python scripts/train_value.py \
  --data_dir "$DAGGER250" \
  --checkpoint_dir "$PISTAR_ROOT/checkpoints/value/assemble_block1_dagger250_s200_f50_accum4_r0" \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_steps 10000 \
  --log_interval 100 \
  --save_interval 2500 \
  --num_workers 0 \
  --wandb_mode disabled \
  --tokenizer_path "$VLM_ROOT/tokenizer.model" \
  --siglip_checkpoint_path "$VLM_ROOT/siglip2-so400m-patch14-224-jax/siglip2_so400m14_224.npz" \
  --gemma_checkpoint_dir "$VLM_ROOT/gemma-3-270m" \
  --freeze_mode all_backbones \
  --load_pretrained
```

Quick terminal-only check:

```bash
cd "$PISTAR_ROOT"

PYTHONPATH=src venv/bin/python scripts/export_vlm_values.py \
  --data_dir "$DAGGER250" \
  --checkpoint_dir "$PISTAR_ROOT/checkpoints/value/assemble_block1_dagger250_s200_f50_accum4_r0" \
  --checkpoint_name step_00010000 \
  --output_path outputs/assemble_block1_dagger250_value_step10000_terminal.parquet \
  --tokenizer_path "$VLM_ROOT/tokenizer.model" \
  --batch_size 4 \
  --num_workers 0 \
  --terminal_only
```

The reported value model separated terminal success/failure approximately as:

```text
success terminal mean ~= -0.415
failure terminal mean ~= -0.586
AUC ~= 0.799
```

You can also export a few full episodes and render side-view value curves:

```bash
PYTHONPATH=src venv/bin/python scripts/export_vlm_values.py \
  --data_dir "$DAGGER250" \
  --checkpoint_dir "$PISTAR_ROOT/checkpoints/value/assemble_block1_dagger250_s200_f50_accum4_r0" \
  --checkpoint_name step_00010000 \
  --output_path outputs/assemble_block1_dagger250_value_step10000_selected_episodes.parquet \
  --tokenizer_path "$VLM_ROOT/tokenizer.model" \
  --batch_size 4 \
  --num_workers 0 \
  --episode_ids 0 1

PYTHONPATH=src venv/bin/python scripts/render_episode_value_video.py \
  --values_path outputs/assemble_block1_dagger250_value_step10000_selected_episodes.parquet \
  --output_dir outputs/value_videos \
  --camera_column side_image \
  --stride 2
```

## 7. Label Advantage

Make a copy before modifying labels:

```bash
test ! -e "$DAGGER250_ADV" && cp -a "$DAGGER250" "$DAGGER250_ADV"
```

Label advantage using the trained value model:

```bash
cd "$PISTAR_ROOT"

HF_DATASETS_CACHE=/tmp/hf_datasets_cache_dagger250_value \
MPLCONFIGDIR=/tmp/matplotlib-pistar \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
PYTHONPATH=src venv/bin/python scripts/label_advantage_from_vlm.py \
  --data_dir "$DAGGER250_ADV" \
  --checkpoint_dir "$PISTAR_ROOT/checkpoints/value/assemble_block1_dagger250_s200_f50_accum4_r0" \
  --checkpoint_name step_00010000 \
  --tokenizer_path "$VLM_ROOT/tokenizer.model" \
  --batch_size 4 \
  --num_workers 0 \
  --lookahead 50 \
  --top_percent 30 \
  --right_wrist_image_col side_image
```

Label rule:

```text
intervention frames -> positive
top 30% non-intervention advantage frames -> positive
remaining rollout frames -> negative
```

The reported run produced:

```text
episodes=250
frames=160410
positive=61236
negative=99174
intervention_frames=18733
positive_intervention=18733
```

## 8. Build PiStar Policy Dataset

Final policy dataset:

```text
demo250_positive + success_dagger200_adv * 3
episodes = 250 + 200 * 3 = 850
```

Convert the 250 demos to PiStar flat format and mark them positive:

```bash
cd "$PISTAR_ROOT"

PYTHONPATH=src venv/bin/python scripts/convert_lerobot_v21_to_pistar_flat.py \
  --source "$DEMO250" \
  --output "$DEMO250_PISTAR" \
  --target-fps 30 \
  --adv-ind positive \
  --intervention 1 \
  --overwrite
```

Keep successful DAgger episodes:

```bash
PYTHONPATH=src venv/bin/python scripts/filter_success_episodes.py \
  --input-root "$DAGGER250_ADV" \
  --output-root "$DAGGER_SUCCESS200" \
  --criterion reward \
  --num-workers 8 \
  --overwrite
```

Merge demos plus three copies of successful DAgger:

```bash
PYTHONPATH=src venv/bin/python scripts/merge_datasets.py \
  --sources \
    "$DEMO250_PISTAR" \
    "$DAGGER_SUCCESS200" \
    "$DAGGER_SUCCESS200" \
    "$DAGGER_SUCCESS200" \
  --output "$POLICY_DATA" \
  --fps 30 \
  --num-workers 8 \
  --overwrite

wc -l "$POLICY_DATA/meta/episodes.jsonl"
```

Expected:

```text
850
```

## 9. Train PiStar0.6

Offline setup:

```bash
cd "$PISTAR_ROOT"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export HF_LEROBOT_HOME=/path/to/lerobot
export OPENPI_DATA_HOME=/path/to/openpi_cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_success200_x3
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
```

Make sure `paligemma_tokenizer.model` exists under `OPENPI_DATA_HOME`. Do not replace it with the VLM Gemma tokenizer.

Compute normalization statistics:

```bash
PYTHONPATH=src venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_star_assemble_blocks_h50_from_pi05 \
  --repo-id piper/assemble_block1_policy_demo250_success200_x3_r1 \
  --local-data-dir "$POLICY_DATA"
```

If training uses `--assets-base-dir /path/to/pistar_assets`, the norm file should be:

```text
/path/to/pistar_assets/pi05_star_assemble_blocks_h50_from_pi05/piper/assemble_block1_policy_demo250_success200_x3_r1/norm_stats.json
```

Train from `pi05_base`. The batch size is global, not per GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTHONPATH=src venv/bin/python scripts/train.py pi05_star_assemble_blocks_h50_from_pi05 \
  --exp-name rollout2_850 \
  --data.repo-id piper/assemble_block1_policy_demo250_success200_x3_r1 \
  --assets-base-dir /path/to/pistar_assets \
  --checkpoint-base-dir /path/to/pistar_runs \
  --weight-loader.params-path "$PI05_BASE_PARAMS" \
  --batch-size 192 \
  --fsdp-devices 6 \
  --num-workers 8 \
  --num-train-steps 10000 \
  --save-interval 2500 \
  --keep-period 2500 \
  --no-wandb-enabled \
  --overwrite
```

## 10. Serve and Evaluate With RTC

Copy the trained checkpoint and matching `norm_stats.json` to the robot machine. The policy server expects norm stats under the checkpoint assets directory. For the current infer config:

```bash
export PISTAR_CKPT=/path/to/checkpoint_step

mkdir -p "$PISTAR_CKPT/assets/piper/assemble_block1_pistar_30hz_3view"
cp /path/to/norm_stats.json \
  "$PISTAR_CKPT/assets/piper/assemble_block1_pistar_30hz_3view/norm_stats.json"
```

Serve the policy:

```bash
cd "$PISTAR_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHONPATH=src venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_star_assemble_blocks_h50_from_pi05_infer \
  --policy.dir "$PISTAR_CKPT" \
  --policy.adv-guidance-beta 1.2
```

Evaluate without saving rollout data:

```bash
cd "$PISTAR_ROOT"

scripts/run_pi05_rtc_success_eval.sh \
  --control-repo-path "$CONTROL_REPO" \
  --server-host localhost \
  --server-port 8000 \
  --task-name "$TASK_NAME" \
  --adv-ind positive \
  --arm-can can0 \
  --arm-name left_arm \
  --state-source joint \
  --cam-head-serial "$HEAD_SERIAL" \
  --cam-side-serial "$SIDE_SERIAL" \
  --cam-wrist-serial "$WRIST_SERIAL" \
  --fps 30 \
  --control-dt 0.033333 \
  --action-horizon 50 \
  --max-step 1500 \
  --num-id 50 \
  --num-position-ood 0 \
  --num-angle-ood 0 \
  --post-reset-sleep 2.0 \
  --rtc-enabled true \
  --rtc-execution-horizon 10 \
  --rtc-max-guidance-weight 10.0 \
  --rtc-prefix-attention-schedule exp \
  --rtc-measure-inference-delay false \
  --rtc-inference-delay-steps 4 \
  --rtc-prefetch-threshold 20 \
  --rtc-worker-sleep 0.005 \
  --rtc-debug false
```

RTC notes:

```text
--rtc-measure-inference-delay false
  Use the fixed delay from --rtc-inference-delay-steps.

--rtc-inference-delay-steps 4
  Good when latency is around 85-140 ms at 30 Hz.
  Try 5 if latency is consistently near 140 ms and motion looks delayed.

--policy.adv-guidance-beta
  beta=0.0 disables advantage guidance and is useful as an ablation.
  beta=1.0 is standard conditional guidance.
  beta>1.0 strengthens positive-advantage guidance, but can increase latency or overshoot.
```

Local beta sweep on the reported checkpoint:

```text
beta=0.0  success=78%
beta=0.5  success=78%
beta=1.0  success=84%
beta=1.1  success=82%
beta=1.2  success=90%
beta=1.3  success=76%
beta=1.5  success=74%
beta=2.0  success=74%
```

Use `beta=1.2` only after confirming it on your robot. A conservative default is `beta=1.0`.

To save evaluation rollout data, use `scripts/run_pi05_rtc_rollout_collect.sh` with the same RTC arguments and a new `--repo-id`.
