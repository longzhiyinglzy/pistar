# PiStar0.6 AssembleBlock1 复现实验流程

这个仓库是基于 [ybpy/pistar](https://github.com/ybpy/pistar) 做的 Piper 单臂三视角装配任务复现版本，任务是：

```text
Pick up the block1 and assemble it.
```

当前最有效的一版：

```text
checkpoint: /home/user/code/pistar/checkpoints/rollout2_850/10000
policy data: demo250_positive + success_dagger200_adv * 3
episodes: 850
init: pi05_base
action: delta joint action training, absolute joint action inference
infer adv_ind: positive
best beta so far: 1.2
```

Beta sweep 结果：

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

建议先复测 `beta=1.2`。如果复测仍然高于 86%，最终用 `1.2`；如果掉回 82%-84%，用更稳的 `1.0`。

## 目录说明

GitHub 只保存代码和流程文档，不保存数据集、checkpoint、输出视频或大模型权重。

不要提交：

```text
checkpoints/
outputs/
.idea/
LeRobot 数据集
Pi0.5 / PiStar checkpoint
VLM / SigLIP / Gemma 权重
```

本地笔记文件 `代码` 只保留在本机，不再上传 GitHub。

## 硬件和相机

三视角 Realsense：

```bash
head=323522063521
side=349222061138
wrist=409122272461
```

机械臂：

```bash
arm_name=left_arm
can_device=can0
state_source=joint
control_dt=0.033333
fps=30
action_horizon=50
```

## 数据路径

本地主要路径：

```bash
# 原始 500 条成功 demo，LeRobot v2.1
/home/user/.cache/huggingface/lerobot/piper/assemble_block1_v21

# 从 500 条均匀抽样得到的 demo250
/home/user/.cache/huggingface/lerobot/piper/assemble_block1_v21_demo250_uniform

# 初始 policy rollout + SpaceMouse DAgger，250 条，200 成功 / 50 失败
/home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_dagger_r0

# DAgger 250 条打好 advantage 之后的数据
/home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_dagger_r0_adv_value_accum4_r0

# demo250 转成 PiStar flat，全部 adv_ind=positive
/home/user/.cache/huggingface/lerobot/piper/assemble_block1_v21_demo250_uniform_pistar_30hz_3view_positive

# 最终 policy 训练集：demo250 + success DAgger200 x3
/home/user/.cache/huggingface/lerobot/piper/assemble_block1_policy_demo250_success200_x3_r1
```

当前本地可用 checkpoint：

```bash
# 初始 pi0.5 policy
/home/user/code/pistar/checkpoints/pistar_AssembleBlock1_250/11000

# value function
/home/user/code/pistar/checkpoints/value/assemble_block1_dagger250_s200_f50_accum4_r0/step_00010000

# 最终 PiStar policy
/home/user/code/pistar/checkpoints/rollout2_850/10000
```

## 1. 从 500 条 Demo 抽 250 条

```bash
cd /home/user/code/pistar

PYTHONPATH=src venv/bin/python scripts/sample_lerobot_episode_subset.py \
  --source /home/user/.cache/huggingface/lerobot/piper/assemble_block1_v21 \
  --output /home/user/.cache/huggingface/lerobot/piper/assemble_block1_v21_demo250_uniform \
  --count 250 \
  --strategy stratified \
  --seed 42 \
  --video-mode copy \
  --overwrite
```

`stratified` 会按 episode 顺序分层抽样，尽量保持原始数据分布。严格实验建议用 `--video-mode copy`，不要软链接。

## 2. 训练初始 Pi0.5 Policy

这一步在 openpi 环境里做，用 demo250 从 `pi05_base` 训练初始 policy。

本地已经用过的初始 policy checkpoint：

```bash
/home/user/code/pistar/checkpoints/pistar_AssembleBlock1_250/11000
```

启动初始 policy server：

```bash
cd /home/user/code/openpi

export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_Piper_AssembleBlock1_inference \
  --policy.dir=/home/user/code/pistar/checkpoints/pistar_AssembleBlock1_250/11000 \
  --port=8000
```

## 3. SpaceMouse DAgger Rollout

目标：

```text
total=250
success=200
failure=50
sticky intervention=true
max_step=1500
```

采集命令：

```bash
cd /home/user/code/pistar

scripts/run_pi05_rtc_rollout_collect.sh \
  --control-repo-path /home/user/code/control_your_robot \
  --server-host localhost \
  --server-port 8000 \
  --repo-id assemble_block1_demo250_dagger_r0 \
  --output-dir /home/user/.cache/huggingface/lerobot/piper \
  --task-name "Pick up the block1 and assemble it." \
  --arm-can can0 \
  --arm-name left_arm \
  --state-source joint \
  --cam-head-serial 323522063521 \
  --cam-side-serial 349222061138 \
  --cam-wrist-serial 409122272461 \
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

按键：

```text
Enter  start
s      save success
f      save failure
r      discard episode
h      home and mark following frames as intervention
q/Esc  quit
```

当前采集结果：

```text
saved_total=250
success=200
failure=50
dagger=64
dagger_rate=25.6%
intervention_frames=18733
discarded=37
```

## 4. 训练 Value Function

value 数据使用 DAgger 250 条，也就是成功 200 + 失败 50。

```bash
cd /home/user/code/pistar

export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_dagger250_value
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHONPATH=src venv/bin/python scripts/train_value.py \
  --data_dir /home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_dagger_r0 \
  --checkpoint_dir /home/user/code/pistar/checkpoints/value/assemble_block1_dagger250_s200_f50_accum4_r0 \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --num_train_steps 10000 \
  --log_interval 100 \
  --save_interval 2500 \
  --num_workers 0 \
  --wandb_mode disabled \
  --tokenizer_path /home/user/hf_models/ybpy/vlm_ckpt/tokenizer.model \
  --siglip_checkpoint_path /home/user/hf_models/ybpy/vlm_ckpt/siglip2-so400m-patch14-224-jax/siglip2_so400m14_224.npz \
  --gemma_checkpoint_dir /home/user/hf_models/ybpy/vlm_ckpt/gemma-3-270m \
  --freeze_mode all_backbones \
  --load_pretrained
```

快速只导出终点 value：

```bash
cd /home/user/code/pistar

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

PYTHONPATH=src venv/bin/python scripts/export_vlm_values.py \
  --data_dir /home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_dagger_r0 \
  --checkpoint_dir /home/user/code/pistar/checkpoints/value/assemble_block1_dagger250_s200_f50_accum4_r0 \
  --checkpoint_name step_00010000 \
  --output_path outputs/assemble_block1_dagger250_value_step10000_terminal.parquet \
  --tokenizer_path /home/user/hf_models/ybpy/vlm_ckpt/tokenizer.model \
  --batch_size 4 \
  --num_workers 0 \
  --terminal_only
```

这版 value function：

```text
success terminal mean ~= -0.415
failure terminal mean ~= -0.586
AUC ~= 0.799
```

## 5. 给 DAgger 数据打 Advantage

先复制一份，不改原始 DAgger 数据：

```bash
cd /home/user/code/pistar

SRC=/home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_dagger_r0
ADV=/home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_dagger_r0_adv_value_accum4_r0

test ! -e "$ADV" && cp -a "$SRC" "$ADV"
```

打 advantage：

```bash
HF_DATASETS_CACHE=/tmp/hf_datasets_cache_dagger250_value \
MPLCONFIGDIR=/tmp/matplotlib-pistar \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache \
PYTHONPATH=src venv/bin/python scripts/label_advantage_from_vlm.py \
  --data_dir /home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_dagger_r0_adv_value_accum4_r0 \
  --checkpoint_dir /home/user/code/pistar/checkpoints/value/assemble_block1_dagger250_s200_f50_accum4_r0 \
  --checkpoint_name step_00010000 \
  --tokenizer_path /home/user/hf_models/ybpy/vlm_ckpt/tokenizer.model \
  --batch_size 4 \
  --num_workers 0 \
  --lookahead 50 \
  --top_percent 30 \
  --right_wrist_image_col side_image
```

实际结果：

```text
episodes=250
frames=160410
positive=61236
negative=99174
intervention_frames=18733
positive_intervention=18733
```

规则：

- SpaceMouse 干预帧强制 `positive`。
- 非干预帧按 VLM value advantage 取 top 30% 为 `positive`。
- 剩下为 `negative`。

## 6. 生成最终 Policy 数据集

最终 policy 数据：

```text
demo250_positive + success_dagger200_adv * 3
episodes = 250 + 200 * 3 = 850
```

转换 demo250：

```bash
cd /home/user/code/pistar

PYTHONPATH=src venv/bin/python scripts/convert_lerobot_v21_to_pistar_flat.py \
  --source /home/user/.cache/huggingface/lerobot/piper/assemble_block1_v21_demo250_uniform \
  --output /home/user/.cache/huggingface/lerobot/piper/assemble_block1_v21_demo250_uniform_pistar_30hz_3view_positive \
  --target-fps 30 \
  --adv-ind positive \
  --intervention 1 \
  --overwrite
```

筛出成功 DAgger 200 条：

```bash
cd /home/user/code/pistar

PYTHONPATH=src venv/bin/python scripts/filter_success_episodes.py \
  --input-root /home/user/.cache/huggingface/lerobot/piper/assemble_block1_demo250_dagger_r0_adv_value_accum4_r0 \
  --output-root /home/user/.cache/huggingface/lerobot/piper/assemble_block1_dagger_success200_adv_r1 \
  --criterion reward \
  --num-workers 8 \
  --overwrite
```

合并并把成功 DAgger 重复 3 次：

```bash
cd /home/user/code/pistar

PYTHONPATH=src venv/bin/python scripts/merge_datasets.py \
  --sources \
    /home/user/.cache/huggingface/lerobot/piper/assemble_block1_v21_demo250_uniform_pistar_30hz_3view_positive \
    /home/user/.cache/huggingface/lerobot/piper/assemble_block1_dagger_success200_adv_r1 \
    /home/user/.cache/huggingface/lerobot/piper/assemble_block1_dagger_success200_adv_r1 \
    /home/user/.cache/huggingface/lerobot/piper/assemble_block1_dagger_success200_adv_r1 \
  --output /home/user/.cache/huggingface/lerobot/piper/assemble_block1_policy_demo250_success200_x3_r1 \
  --fps 30 \
  --num-workers 8 \
  --overwrite

wc -l /home/user/.cache/huggingface/lerobot/piper/assemble_block1_policy_demo250_success200_x3_r1/meta/episodes.jsonl
```

期望输出：

```text
850
```

## 7. 内网离线 Docker 准备

进入容器：

```bash
docker exec -it pistar_train /bin/bash
cd /home/user/code/pistar
```

离线环境变量：

```bash
cd /home/user/code/pistar

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export HF_LEROBOT_HOME=/home/user/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/root/.cache/openpi
export HF_DATASETS_CACHE=/data/cache/hf_datasets_success200_x3
export JAX_COMPILATION_CACHE_DIR=/data/cache/jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu

mkdir -p "$HF_DATASETS_CACHE" "$JAX_COMPILATION_CACHE_DIR"
```

如果数据放在 `/data/lerobot/piper`，注册到 LeRobot 默认路径：

```bash
mkdir -p /home/user/.cache/huggingface/lerobot/piper

ln -sfn \
  /data/lerobot/piper/assemble_block1_policy_demo250_success200_x3_r1 \
  /home/user/.cache/huggingface/lerobot/piper/assemble_block1_policy_demo250_success200_x3_r1
```

确认 Pi0.5 tokenizer：

```bash
find /root/.cache/openpi /home/user/.cache/openpi /workspaces /data \
  -type f -name paligemma_tokenizer.model 2>/dev/null
```

不要用 VLM 的 Gemma tokenizer 替代 `paligemma_tokenizer.model`。

## 8. 确认 Delta Action 配置

训练和推理必须看到：

```text
inputs:
  PiperInputs()
  DeltaActions(mask=(True, True, True, True, True, True, False))
outputs:
  AbsoluteActions(mask=(True, True, True, True, True, True, False))
  PiperOutputs()
```

检查命令：

```bash
cd /home/user/code/pistar

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

当前保持：

```text
discrete_state_input=False
extra_delta_transform=True
```

## 9. 计算 Norm

内网 Docker：

```bash
cd /home/user/code/pistar

PYTHONPATH=src venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_star_assemble_blocks_h50_from_pi05 \
  --repo-id piper/assemble_block1_policy_demo250_success200_x3_r1 \
  --local-data-dir /data/lerobot/piper/assemble_block1_policy_demo250_success200_x3_r1
```

训练时如果使用：

```bash
--assets-base-dir /data/pistar_assets
```

则 norm 应该在：

```bash
/data/pistar_assets/pi05_star_assemble_blocks_h50_from_pi05/piper/assemble_block1_policy_demo250_success200_x3_r1/norm_stats.json
```

如果脚本默认写到了 repo assets，就复制过去：

```bash
mkdir -p /data/pistar_assets/pi05_star_assemble_blocks_h50_from_pi05/piper/assemble_block1_policy_demo250_success200_x3_r1

cp \
  /home/user/code/pistar/assets/pi05_star_assemble_blocks_h50_from_pi05/piper/assemble_block1_policy_demo250_success200_x3_r1/norm_stats.json \
  /data/pistar_assets/pi05_star_assemble_blocks_h50_from_pi05/piper/assemble_block1_policy_demo250_success200_x3_r1/norm_stats.json
```

## 10. 内网训练 PiStar Policy

6 张 H20，`batch_size=192` 是总 batch，不是单卡 batch。

```bash
cd /home/user/code/pistar

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export HF_LEROBOT_HOME=/home/user/.cache/huggingface/lerobot
export OPENPI_DATA_HOME=/root/.cache/openpi
export HF_DATASETS_CACHE=/data/cache/hf_datasets_success200_x3
export JAX_COMPILATION_CACHE_DIR=/data/cache/jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu

mkdir -p "$HF_DATASETS_CACHE" "$JAX_COMPILATION_CACHE_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
PYTHONPATH=src venv/bin/python scripts/train.py pi05_star_assemble_blocks_h50_from_pi05 \
  --exp-name rollout2_850 \
  --data.repo-id piper/assemble_block1_policy_demo250_success200_x3_r1 \
  --assets-base-dir /data/pistar_assets \
  --checkpoint-base-dir /data/pistar_runs \
  --weight-loader.params-path /mnt/pi05_base/params \
  --batch-size 192 \
  --fsdp-devices 6 \
  --num-workers 8 \
  --num-train-steps 10000 \
  --save-interval 2500 \
  --keep-period 2500 \
  --no-wandb-enabled \
  --overwrite
```

这一步从 `/mnt/pi05_base/params` 开始，也就是从 `pi05_base` 训练，不是从之前 40000 step 的 full finetune 开始。

## 11. 本地推理

当前本地 checkpoint：

```bash
/home/user/code/pistar/checkpoints/rollout2_850/10000
```

确保 checkpoint 下有 norm：

```bash
cd /home/user/code/pistar

mkdir -p checkpoints/rollout2_850/10000/assets/piper/assemble_block1_pistar_30hz_3view

cp checkpoints/rollout2_850/norm_stats.json \
  checkpoints/rollout2_850/10000/assets/piper/assemble_block1_pistar_30hz_3view/norm_stats.json
```

启动 server，当前推荐先用 `beta=1.2`：

```bash
cd /home/user/code/pistar

export XLA_PYTHON_CLIENT_PREALLOCATE=false

PYTHONPATH=src venv/bin/python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config=pi05_star_assemble_blocks_h50_from_pi05_infer \
  --policy.dir=/home/user/code/pistar/checkpoints/rollout2_850/10000 \
  --policy.adv-guidance-beta 1.2
```

保守部署可以用：

```bash
--policy.adv-guidance-beta 1.0
```

## 12. 本地成功率测试

只评估，不保存数据：

```bash
cd /home/user/code/pistar

scripts/run_pi05_rtc_success_eval.sh \
  --control-repo-path /home/user/code/control_your_robot \
  --server-host localhost \
  --server-port 8000 \
  --task-name "Pick up the block1 and assemble it." \
  --adv-ind positive \
  --arm-can can0 \
  --arm-name left_arm \
  --state-source joint \
  --cam-head-serial 323522063521 \
  --cam-side-serial 349222061138 \
  --cam-wrist-serial 409122272461 \
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

RTC 建议：

- `beta=1.0` 通常 latency 约 85 ms，`--rtc-inference-delay-steps 4` 合适。
- `beta>1.0` 可能需要算 unconditional 分支，latency 可能升高；如果稳定 140 ms，可以试 `--rtc-inference-delay-steps 5`。
- 当前 `beta=1.2` 成功率最高，先用 delay 4 测；如果动作抖，再调 delay 5。

## 13. 保存式 Rollout

如果要保存 50 条评估数据：

```bash
cd /home/user/code/pistar

scripts/run_pi05_rtc_rollout_collect.sh \
  --control-repo-path /home/user/code/control_your_robot \
  --server-host localhost \
  --server-port 8000 \
  --repo-id assemble_block1_rollout2_850_beta12_eval50 \
  --output-dir /home/user/.cache/huggingface/lerobot/piper \
  --task-name "Pick up the block1 and assemble it." \
  --adv-ind positive \
  --arm-can can0 \
  --arm-name left_arm \
  --state-source joint \
  --cam-head-serial 323522063521 \
  --cam-side-serial 349222061138 \
  --cam-wrist-serial 409122272461 \
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

## 14. 磁盘清理

低风险缓存：

```bash
rm -rf /home/user/.local/share/Trash/files/*
rm -rf /home/user/.local/share/Trash/info/*
rm -rf /home/user/.cache/huggingface/datasets
rm -rf /home/user/.cache/pip
rm -rf /home/user/.cache/vscode-cpptools/ipch
```

如果只推理、不再 resume，可以删除旧 checkpoint 的 `train_state`：

```bash
rm -rf /home/user/code/pistar/checkpoints/rollout2/10000/train_state
rm -rf /home/user/code/pistar/checkpoints/pistar_AssembleBlock1_250/11000/train_state
```

不要删：

```bash
/home/user/code/pistar/checkpoints/rollout2_850/10000/params
/home/user/code/pistar/checkpoints/rollout2_850/10000/assets
/home/user/code/pistar/checkpoints/rollout2_850/norm_stats.json
```
