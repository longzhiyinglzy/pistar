# PiStar0.6 Single-Arm Piper SpaceMouse Reproduction

This repository extends [ybpy/pistar](https://github.com/ybpy/pistar) with a reproducible single-arm Piper pipeline for SpaceMouse demonstration collection, DAgger intervention, value learning, advantage labeling, PiStar0.6 training, and RTC evaluation.

Reference protocol:

```text
base policy: pi0.5 base
demo data: 250 successful LeRobot v2.1 demos
DAgger data: 250 rollout episodes, 200 success + 50 failure
policy data: demo250_positive + success_dagger200_adv repeated 3 times
policy episodes: 850
control rate: 30 Hz
action horizon: 50
guidance beta: selected by validation
```

## Environment Setup

Create the robot-side environment:

```bash
git clone https://github.com/longzhiyinglzy/pistar.git
cd pistar

conda create -n pistar python=3.11 -y
conda activate pistar
python -m pip install --upgrade pip

python -m pip install -e control_your_robot/src/robot/piper_sdk
python -m pip install -e control_your_robot
python -m pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
```

The LeRobot revision provides the 0.1.0 API required by the v2.1 converter. Install `ffmpeg` through the operating-system package manager for video-backed datasets.

Create the PiStar training environment with `uv`:

```bash
python -m pip install uv
UV_PROJECT_ENVIRONMENT=venv uv sync --frozen
```

## Path Configuration

```bash
export PISTAR_ROOT=/path/to/pistar
export OPENPI_ROOT=/path/to/openpi
export CONTROL_REPO=$PISTAR_ROOT/control_your_robot
export HF_LEROBOT_HOME=/path/to/lerobot
export LEROBOT_ROOT=$HF_LEROBOT_HOME/piper
export HDF5_ROOT=/path/to/piper_datasets
export TASK_ID=your_task
export HDF5_TASK_NAME=${TASK_ID}_hdf5
export RAW_HDF5_DIR=$HDF5_ROOT/$HDF5_TASK_NAME
export DEMO_REPO_ID=piper/${TASK_ID}_v21
export DEMO250_REPO_ID=piper/${TASK_ID}_demo250

export DEMO_V21=$LEROBOT_ROOT/${TASK_ID}_v21
export DEMO250=$LEROBOT_ROOT/${TASK_ID}_demo250
export DAGGER250=$LEROBOT_ROOT/${TASK_ID}_dagger250
export DAGGER250_ADV=$LEROBOT_ROOT/${TASK_ID}_dagger250_adv
export DEMO250_PISTAR=$LEROBOT_ROOT/${TASK_ID}_demo250_pistar_positive
export DAGGER_SUCCESS200=$LEROBOT_ROOT/${TASK_ID}_dagger_success200_adv
export POLICY_DATA=$LEROBOT_ROOT/${TASK_ID}_policy_demo250_success200_x3
export POLICY_REPO_ID=piper/${TASK_ID}_policy_demo250_success200_x3
export VALUE_CKPT=$PISTAR_ROOT/checkpoints/value/${TASK_ID}_dagger250

export VLM_ROOT=/path/to/ybpy/vlm_ckpt
export PI05_BASE_PARAMS=/path/to/pi05_base/params
export PI05_TRAIN_CONFIG=your_pi05_train_config
export PI05_INFER_CONFIG=your_pi05_inference_config

export HEAD_SERIAL=<head_camera_serial>
export SIDE_SERIAL=<side_camera_serial>
export WRIST_SERIAL=<wrist_camera_serial>
export TASK_NAME="your task instruction"
export NUM_DEMOS=<number_of_successful_demo_episodes>
```

## Action Representation

The Piper PiStar config uses `extra_delta_transform=True`.

```text
dataset action columns: absolute target joint positions
model training target: delta joint action for joints 1-6
gripper target: absolute
deployment command: absolute target joint position after output transform
```

`DeltaActions(mask=(True, True, True, True, True, True, False))` converts joints 1-6 to delta targets during training. `AbsoluteActions(...)` reconstructs absolute joint targets for deployment. The gripper remains absolute.

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

Reference hardware:

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

Set camera serials either through command-line arguments in the rollout/eval scripts or in `control_your_robot/my_robot/camera_config.py`.

Activate the Piper CAN interface after each reboot or USB-CAN reconnect:

```bash
cd "$PISTAR_ROOT/control_your_robot"
bash src/robot/piper_sdk/piper_sdk/can_activate.sh can0 1000000
ip -details link show can0
```

The interface must report `state UP`, `can state ERROR-ACTIVE`, and `bitrate 1000000`.

## 2. Collect HDF5 / LeRobot v2.1 Demos With SpaceMouse

```text
SpaceMouse -> Piper HDF5 demos -> LeRobot v2.1
```

```bash
conda activate pistar
cd "$PISTAR_ROOT/control_your_robot"

python example/collect/collect_piper.py \
  --save-path "$HDF5_ROOT" \
  --task-name "$HDF5_TASK_NAME" \
  --num-episode "$NUM_DEMOS" \
  --arm-can can0 \
  --motion-speed-percent 10 \
  --reset-speed-percent 10
```

The collector discovers 3Dconnexion devices automatically. Use `--spacemouse-device-path /dev/input/eventX` to select a specific device.

Keyboard controls:

```text
Enter  start episode, then save and stop current episode
r      discard current episode and retry the same episode index
h      home the arm and continue recording the same episode
```

The control and camera-aligned sampling loops run at 200 Hz and 30 Hz, respectively. HDF5 output includes synchronized state and action buffers.

## 3. Convert HDF5 to LeRobot v2.1

Convert Piper HDF5 joint-pose demos to LeRobot v2.1:

```bash
conda activate pistar
cd "$PISTAR_ROOT/control_your_robot"
export HF_LEROBOT_HOME=/path/to/lerobot

python scripts/convert2openpi_piper_jointpose.py \
  --raw-dir "$RAW_HDF5_DIR" \
  --repo-id "$DEMO_REPO_ID" \
  --task "$TASK_NAME" \
  --mode video
```

The converter expects three real camera streams in the HDF5 files:

```text
slave_cam_head/color  -> observation.images.cam_head
slave_cam_side/color  -> observation.images.cam_side
slave_cam_wrist/color -> observation.images.cam_wrist
```

The converted dataset contains 7-D `observation.state` and `action`: six joints and one gripper value. The default `action_mode="next_state"` stores the next observed joint state as the action target.

The reference protocol uses a stratified 250-episode demonstration subset:

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

`--video-mode copy` produces a standalone subset.

## 4. Train Initial Pi0.5 Policy

Train the initial policy from `pi05_base` on `DEMO250` in OpenPI.

```bash
cd "$OPENPI_ROOT"

uv run scripts/compute_norm_stats.py \
  --config-name "$PI05_TRAIN_CONFIG" \
  --repo-id "$DEMO250_REPO_ID" \
  --local-data-dir "$DEMO250"

uv run scripts/train.py "$PI05_TRAIN_CONFIG" \
  --exp-name "${TASK_ID}_pi05" \
  --data.repo-id "$DEMO250_REPO_ID" \
  --weight-loader.params-path "$PI05_BASE_PARAMS" \
  --num-train-steps 40000 \
  --save-interval 5000 \
  --keep-period 5000 \
  --no-wandb-enabled \
  --overwrite
```

Serve the trained initial policy for rollout:

```bash
cd "$OPENPI_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config="$PI05_INFER_CONFIG" \
  --policy.dir /path/to/initial_pi05_demo250_checkpoint \
  --port 8000
```

The initial policy generates DAgger rollouts. PiStar training is initialized independently from `pi05_base`.

## 5. Rollout Initial Policy With SpaceMouse DAgger

Collect DAgger rollouts with sticky SpaceMouse intervention:

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

Controls:

```text
Enter  start episode
s      save success
f      save failure
r      discard
h      home and mark following frames as intervention
q/Esc  quit
```

## 6. Train VLM Value Function

Train the value model on all successful and failed DAgger episodes.

```bash
cd "$PISTAR_ROOT"

export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_dagger250_value
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHONPATH=src venv/bin/python scripts/train_value.py \
  --data_dir "$DAGGER250" \
  --checkpoint_dir "$VALUE_CKPT" \
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

## 7. Label Advantage

Create a labeled working copy:

```bash
test ! -e "$DAGGER250_ADV" && cp -a "$DAGGER250" "$DAGGER250_ADV"
```

Label advantage with the trained value model:

```bash
cd "$PISTAR_ROOT"

HF_DATASETS_CACHE=/tmp/hf_datasets_cache_dagger250_value \
MPLCONFIGDIR=/tmp/matplotlib-pistar \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
PYTHONPATH=src venv/bin/python scripts/label_advantage_from_vlm.py \
  --data_dir "$DAGGER250_ADV" \
  --checkpoint_dir "$VALUE_CKPT" \
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

Filter successful DAgger episodes:

```bash
PYTHONPATH=src venv/bin/python scripts/filter_success_episodes.py \
  --input-root "$DAGGER250_ADV" \
  --output-root "$DAGGER_SUCCESS200" \
  --criterion reward \
  --num-workers 8 \
  --overwrite
```

Merge demonstrations with three copies of successful DAgger data:

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

The reference protocol produces 850 episodes.

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

`paligemma_tokenizer.model` must exist under `OPENPI_DATA_HOME`; it is distinct from the value-model Gemma tokenizer.

Compute normalization statistics:

```bash
PYTHONPATH=src venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_star_piper_h50_from_pi05 \
  --repo-id "$POLICY_REPO_ID" \
  --local-data-dir "$POLICY_DATA"
```

If training uses `--assets-base-dir /path/to/pistar_assets`, the norm file should be:

```text
/path/to/pistar_assets/pi05_star_piper_h50_from_pi05/$POLICY_REPO_ID/norm_stats.json
```

Train from `pi05_base`; `batch-size` is global across FSDP devices:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTHONPATH=src venv/bin/python scripts/train.py pi05_star_piper_h50_from_pi05 \
  --exp-name "${TASK_ID}_pistar" \
  --data.repo-id "$POLICY_REPO_ID" \
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

Copy the matching normalization statistics into the checkpoint assets directory:

```bash
export PISTAR_CKPT=/path/to/checkpoint_step
export NORM_STATS=/path/to/norm_stats.json

mkdir -p "$PISTAR_CKPT/assets/$POLICY_REPO_ID"
cp "$NORM_STATS" "$PISTAR_CKPT/assets/$POLICY_REPO_ID/norm_stats.json"
```

Serve the policy:

```bash
cd "$PISTAR_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHONPATH=src venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_star_piper_h50_from_pi05_infer \
  --policy.dir "$PISTAR_CKPT" \
  --policy.asset-id "$POLICY_REPO_ID" \
  --policy.adv-guidance-beta 1.0
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

`beta=0` disables classifier-free advantage guidance, `beta=1` uses the positive-conditioned policy, and `beta>1` increases guidance strength. The RTC delay must match policy-server latency. Measurements on an RTX 4090 were approximately 85 ms with `beta=1` (`delay_steps=4`) and 140 ms with `beta!=1` (`delay_steps=5`). Guidance strength is selected on a held-out real-robot validation set.
