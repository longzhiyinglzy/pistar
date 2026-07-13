# PiStar0.6 Piper AssembleBlock1 Reproduction

This repository is an experimental real-robot reproduction of [ybpy/pistar](https://github.com/ybpy/pistar) for a Piper single-arm block assembly task with three RealSense views.

Task prompt:

```text
Pick up the block1 and assemble it.
```

High-level result from the current run:

```text
base policy: pi0.5 base
policy data: demo250_positive + success_dagger200_adv * 3
policy episodes: 850
action horizon: 50
control rate: 30 Hz
best guidance beta observed: 1.2
```

Beta sweep on the local robot:

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

Use `beta=1.2` only after re-testing it on your setup. A conservative default is `beta=1.0`.

## Repository Policy

This repository tracks code and reproducible workflow documentation only.

Do not commit:

```text
checkpoints/
outputs/
.idea/
LeRobot datasets
Pi0.5 / PiStar checkpoints
VLM / SigLIP / Gemma weights
```

## Action Representation

This setup uses `extra_delta_transform=True` for the Piper PiStar config.

The important detail is:

```text
dataset action columns: absolute target joint positions
model training target: delta joint action for the first 6 joints
gripper target: absolute
policy server output in this repo: absolute target joint positions for the controller
```

In code, `DeltaActions(mask=(True, True, True, True, True, True, False))` converts absolute joint targets into delta targets for training. During inference, `AbsoluteActions(...)` adds the predicted delta back to the current state before sending actions to the Piper control script.

So the model itself learns relative joint deltas, but the default deployment wrapper emits absolute target joint positions because the provided Piper controller scripts call `set_joint(...)` or `set_position(...)` with target positions. If your controller consumes delta commands directly, remove or replace the `AbsoluteActions` output transform and update the deployment script accordingly.

You can verify the active transforms with:

```bash
cd "$PISTAR_ROOT"

PYTHONPATH=src venv/bin/python - <<'PY'
from openpi.training.config import get_config

for name in [
    "pi05_star_assemble_blocks_h50_from_pi05",
    "pi05_star_assemble_blocks_h50_from_pi05_infer",
]:
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

Expected Piper PiStar transform summary:

```text
inputs:
  PiperInputs()
  DeltaActions(mask=(True, True, True, True, True, True, False))
outputs:
  AbsoluteActions(mask=(True, True, True, True, True, True, False))
  PiperOutputs()
```

## Environment Variables

Set these paths for your machine before running the commands below:

```bash
export PISTAR_ROOT=/path/to/pistar
export OPENPI_ROOT=/path/to/openpi
export CONTROL_REPO=/path/to/control_your_robot
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

For the experiment reported above, the three camera roles are:

```text
head: front/global view
side: side view
wrist: wrist-mounted view
```

## Dataset Fields

The PiStar flat LeRobot datasets used here contain:

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

The Piper config maps them into model views as:

```text
image       -> cam_high
wrist_image -> cam_wrist
side_image  -> cam_wrist1
```

## 1. Sample 250 Demo Episodes

Start from a successful 500-episode LeRobot v2.1 demo dataset and sample 250 episodes uniformly over episode order:

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

Use `--video-mode copy` for a strict standalone dataset. Use symlinks only for quick local experiments.

## 2. Train the Initial Pi0.5 Policy

Train this in your OpenPI environment, not in the PiStar environment. The initial policy is trained from `pi05_base` using `DEMO250`.

Example serve command after training:

```bash
cd "$OPENPI_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_Piper_AssembleBlock1_inference \
  --policy.dir=/path/to/initial_pi05_demo250_checkpoint \
  --port=8000
```

Keep this policy server running for the DAgger rollout step.

## 3. Collect DAgger Rollout With SpaceMouse

Target collection:

```text
total episodes: 250
success episodes: 200
failure episodes: 50
sticky intervention: enabled
max frames per episode: 1500
```

Run from the PiStar repo:

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

Keyboard controls:

```text
Enter  start episode
s      save success
f      save failure
r      discard episode
h      home and mark following frames as intervention
q/Esc  quit
```

The run reported here collected:

```text
saved_total=250
success=200
failure=50
dagger=64
dagger_rate=25.6%
intervention_frames=18733
discarded=37
```

## 4. Train the VLM Value Function

Train the value model on the DAgger 250 episodes, including both success and failure episodes.

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

Optional terminal-only value export:

```bash
cd "$PISTAR_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

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

The reported value model had:

```text
success terminal mean ~= -0.415
failure terminal mean ~= -0.586
AUC ~= 0.799
```

## 5. Label Advantage on DAgger Data

Copy the original DAgger dataset first:

```bash
test ! -e "$DAGGER250_ADV" && cp -a "$DAGGER250" "$DAGGER250_ADV"
```

Then label advantage:

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

Rule used:

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

## 6. Build the Policy Dataset

Final policy data:

```text
demo250_positive + success_dagger200_adv * 3
episodes = 250 + 200 * 3 = 850
```

Convert the sampled demo data into PiStar flat format and mark all demo frames as positive:

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

Keep only successful DAgger episodes:

```bash
cd "$PISTAR_ROOT"

PYTHONPATH=src venv/bin/python scripts/filter_success_episodes.py \
  --input-root "$DAGGER250_ADV" \
  --output-root "$DAGGER_SUCCESS200" \
  --criterion reward \
  --num-workers 8 \
  --overwrite
```

Merge the final policy dataset:

```bash
cd "$PISTAR_ROOT"

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

## 7. Offline Training Setup

In an offline training container, set:

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

mkdir -p "$HF_DATASETS_CACHE" "$JAX_COMPILATION_CACHE_DIR"
```

Make sure `paligemma_tokenizer.model` exists under `OPENPI_DATA_HOME`. Do not replace it with the VLM Gemma tokenizer.

## 8. Compute Normalization Statistics

```bash
cd "$PISTAR_ROOT"

PYTHONPATH=src venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_star_assemble_blocks_h50_from_pi05 \
  --repo-id piper/assemble_block1_policy_demo250_success200_x3_r1 \
  --local-data-dir "$POLICY_DATA"
```

If you train with:

```bash
--assets-base-dir /path/to/pistar_assets
```

then `norm_stats.json` should be under:

```bash
/path/to/pistar_assets/pi05_star_assemble_blocks_h50_from_pi05/piper/assemble_block1_policy_demo250_success200_x3_r1/norm_stats.json
```

## 9. Train the PiStar Policy

Example for six H20 GPUs. `--batch-size 192` is the global batch size, not per-GPU batch size.

```bash
cd "$PISTAR_ROOT"

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

This command initializes from `pi05_base`, not from a task-specific full fine-tuned pi0.5 checkpoint.

## 10. Serve the Trained PiStar Policy

Copy the trained checkpoint and matching `norm_stats.json` to the robot machine. If your infer config expects the default Piper asset id, place norm stats under:

```bash
$PISTAR_CKPT/assets/piper/assemble_block1_pistar_30hz_3view/norm_stats.json
```

Serve with:

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

Use `--policy.adv-guidance-beta 1.0` for the safer baseline.

## 11. Evaluate Success Rate Without Saving Data

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

- `beta=1.0` usually keeps inference latency close to the plain Pi0.5 baseline.
- `beta>1.0` may require the unconditional branch and increase inference latency.
- If latency is stable around 140 ms, try `--rtc-inference-delay-steps 5`.
- For the reported `beta=1.2` run, delay 4 was tested first.

## 12. Save Evaluation Rollout Data

```bash
cd "$PISTAR_ROOT"

scripts/run_pi05_rtc_rollout_collect.sh \
  --control-repo-path "$CONTROL_REPO" \
  --server-host localhost \
  --server-port 8000 \
  --repo-id assemble_block1_rollout2_850_beta12_eval50 \
  --output-dir "$LEROBOT_ROOT" \
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
  --num-episode 50 \
  --max-step 1500 \
  --save-adv-ind positive \
  --enter-label discard \
  --min-save-frames 30 \
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

## Forking and Upstream

For a clean public release, a GitHub fork of `ybpy/pistar` is recommended because it shows provenance and makes upstream sync easier.

If this repository was not created with GitHub's Fork button, GitHub will not display "forked from ybpy/pistar" automatically. You have two practical options:

1. Keep this repository as an independent derivative and clearly credit upstream in the README.
2. Create a new fork from `ybpy/pistar`, then push these commits to a branch or to the fork's main branch.

If you want the GitHub UI to show the fork relationship, choose option 2. If you only need a reproducible public repository, option 1 is enough.
