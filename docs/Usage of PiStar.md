# Usage of PiStar

## Base Environment Setup

Using uv to manage virtual environment. 

```bash
git clone https://github.com/ybpy/pistar.git

git submodule update --init --recursive

uv venv --python 3.11.9 /path/to/create/pistar/venv

source /path/to/your/pistar/venv/bin/activate

cd /path/to/pistar

GIT_LFS_SKIP_SMUDGE=1 uv sync --active

GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv pip install -r pistar_requirements.txt
```

## Fine-Tuning

Before executing following command, make sure the pistar venv have been activated. 

### 1. Convert your data to a LeRobot dataset

```bash
python examples/libero/pistar_data_processing_optimized.py \
  --data_dir /path/to/modified_libero_rlds \
  --default_adv_ind positive
or 
python -u examples/libero/pistar_data_processing_optimized.py ...
```

### 2. Compute the normalization statistics for the training data

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/compute_norm_stats.py --config-name pi05_star_libero
```

### 3. Run training

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_star_libero --exp-name=my_experiment --overwrite
```

You can use `--resume` to replace `--overwrite` in above command to restore latest checkpoint for continuation training. 

## Evaluation and Rollout

### 1. Client

Create virtual environment for Simulation environment first.
```bash
# Create virtual environment
uv venv --python 3.10 /path/to/create/libero_rollout/venv
source /path/to/your/libero_rollout/venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \ 
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip install --no-deps git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
# To insure installing these package in your libero venv, you can add --python /path/to/your/libero/venv/bin/python behind above command, like uv pip install -e third_party/libero --python /path/to/your/libero/venv/bin/python
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

For evaluation:
```bash
# Run the simulation
# Recommend using egl for rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python examples/libero/main.py --args.adv_ind_input positive
# If you have egl errors, fix by running the following command
sudo -E apt-get update
sudo -E apt-get install -y libegl1 libgl1 libglvnd0 libgles2 libdrm2 libgbm1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python examples/libero/main.py --args.adv_ind_input positive
# If failed, try the following command to use glx for rendering
export MUJOCO_GL=glx
xvfb-run -a python examples/libero/main.py --args.adv_ind_input positive
```

For rollout in simulator:
```bash
# Run the simulation
# Recommend using egl for rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python examples/libero/main.py --args.adv_ind_input positive --args.num_trials_per_task 5 --args.save_lerobot_rollout --args.rollout_overwrite
# If you have egl errors, fix by running the following command
sudo -E apt-get update
sudo -E apt-get install -y libegl1 libgl1 libglvnd0 libgles2 libdrm2 libgbm1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python examples/libero/main.py --args.adv_ind_input positive --args.num_trials_per_task 5 --args.save_lerobot_rollout --args.rollout_overwrite
# If failed, try the following command to use glx for rendering
export MUJOCO_GL=glx
xvfb-run -a python examples/libero/main.py --args.adv_ind_input positive --args.num_trials_per_task 5 --args.save_lerobot_rollout --args.rollout_overwrite
```

### 2. Server

```bash
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_star_libero_infer \
  --policy.dir=checkpoints/pi05_star_libero/my_experiment/10000
```

## World Model for PiStar

### World model environment setup

Install additional dependencies for world model in pistar venv:

```bash
source /path/to/your/pistar/venv/bin/activate
pip install -r wm_requirements.txt
```

### Checkpoint and Dataset

| ckpt or dataset                                              | intro                                |
| ------------------------------------------------------------ | ------------------------------------ |
| [openai/clip-vit-base-patch32 · Hugging Face](https://huggingface.co/openai/clip-vit-base-patch32) | CLIP text and image encoder          |
| [stabilityai/stable-video-diffusion-img2vid · Hugging Face](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) | pretrained SVD video diffusion model |
| [yifengzhu-hf/LIBERO-datasets · Datasets at Hugging Face](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) | LIBERO dataset                       |

### Training

#### 1. Prepare dataset

For common libero dataset in lerobot format or rlds format are regenerated from original libero dataset filtering out failure episodes while world model should be trained on some failure data to be not over-optimistic, so we first regenerate libero dataset, just filtering out no-op frames and flipping the image, finally transform it to lerobot format.

1. Regenerate libero dataset

   ```bash
   cd /path/to/pistar
   # do that for every task suite dataset
   python examples/libero/regenerate_libero_dataset.py \
     --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
     --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
     --libero_target_dir <PATH TO TARGET DIR>
   ```

2. Build rlds format dataset

   ```bash
   # parallel with pistar
   git clone https://github.com/ybpy/rlds_dataset_builder.git
   cd rlds_dataset_builder
   ```

   `cd` to corresponding folder and modify the path in `_split_paths()` function at the end of `*_dataset_builder.py`, then run

   ```bash
   tfds build --overwrite
   ```

   output rlds dataset will be in `~/tensorflow_datasets`.

3. Build lerobot format dataset

   ```bash
   cd /path/to/pistar
   python examples/libero/wm_data_processing.py \
     --data_dir /path/to/your/LIBERO_no_noops_rlds \
     --overwrite True
   ```

Since the video diffusion model are run in latent space of image encoder, we need to extract the latent sapce of the video to improve training efficiency. You can run the following command to extract latent in parallel:

```bash
python scripts/extract_latent.py \
  --lerobot_root /path/to/your/lerobot/dataset \
  --output_path /path/to/pistar/dataset/libero_wm \
  --svd_path /path/to/model/stable-video-diffusion-img2vid \
  --overwrite true
```

Video latent dataset will be in `/path/to/pistar/dataset`.

After extract the video latent, we can prepare dataset meta information, which create a json file include all items and calculate the normalization of states and actions, which are required during training.

```bash
python3 scripts/create_meta_info.py \
  --dataset_output_path /path/to/pistar/dataset/libero_wm \
  --dataset_name libero_wm \
  --meta_output_root /path/to/pistar/dataset_meta_info
```

Dataset meta info will be in `/path/to/pistar/dataset_meta_info`.

#### 2. Launch training

After prepare the datasets, you can launch training.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_wm.py
```

### Eval by replaying

We start from an initial observation sampled from the recorded trajectories and then generate long trajectories by replaying the recorded actions.

```bash
python3 scripts/rollout_replay_traj.py --episode_id 99 --mode val
# or
python3 scripts/rollout_replay_traj.py \
  --episode_id 0 \
  --mode train \
  --max_replay_steps 20
```

