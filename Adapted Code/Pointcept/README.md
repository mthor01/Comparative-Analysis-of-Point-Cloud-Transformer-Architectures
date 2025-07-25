The following instructions are adapted from the original Pointcept repository.
They include all information needed to run semantic segmentation training and 
evaluation on the S3DIS dataset for PTv1, PTv3 and OctFormer.

### Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above.
- PyTorch: 1.10.0 and above.

### Conda Environment
- **Method 1**: Utilize conda `environment.yml` to create a new environment with one line code:
  ```bash
  # Create and activate conda environment named as 'pointcept-torch2.5.0-cu12.4'
  # cuda: 12.4, pytorch: 2.5.0

  # run `unset CUDA_PATH` if you have installed cuda in your local environment
  conda env create -f environment.yml --verbose
  conda activate pointcept-torch2.5.0-cu12.4
  ```

- **Method 2**: Use our pre-built Docker image and refer to the supported tags [here](https://hub.docker.com/repository/docker/pointcept/pointcept/general). Quickly verify the Docker image on your local machine with the following command:
  ```bash
  docker run --gpus all -it --rm pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel bash
  git clone https://github.com/facebookresearch/sonata
  cd sonata
  export PYTHONPATH=./ && python demo/0_pca.py
  # Ignore the GUI error, we cannot expect a container to have its GUI, right?
  ```

- **Method 3**: Manually create a conda environment:
  ```bash
  conda create -n pointcept python=3.10 -y
  conda activate pointcept
  
  # (Optional) If no CUDA installed
  conda install nvidia/label/cuda-12.4.1::cuda conda-forge::cudnn conda-forge::gcc=13.2 conda-forge::gxx=13.2 -y
  
  conda install ninja -y
  # Choose version you want here: https://pytorch.org/get-started/previous-versions/
  conda install pytorch==2.5.0 torchvision==0.13.1 torchaudio==0.20.0 pytorch-cuda=12.4 -c pytorch -y
  conda install h5py pyyaml -c anaconda -y
  conda install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
  conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
  pip install torch-geometric

  # spconv (SparseUNet)
  # refer https://github.com/traveller59/spconv
  pip install spconv-cu124

  # PPT (clip)
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git

  # PTv1 & PTv2 or precise eval
  cd libs/pointops
  # usual
  python setup.py install
  # docker & multi GPU arch
  TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
  # e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
  TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
  cd ../..

  # Open3D (visualization, optional)
  pip install open3d
  ```
## Data Preparation

### S3DIS

- Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the `Stanford3dDataset_v1.2.zip` file and unzip it.
- Fix error in `Area_5/office_19/Annotations/ceiling` Line 323474 (103.0�0000 => 103.000000).
- (Optional) Download Full 2D-3D S3DIS dataset (no XYZ) from [here](https://github.com/alexsax/2D-3D-Semantics) for parsing normal.
- Run preprocessing code for S3DIS as follows:

  ```bash
  # S3DIS_DIR: the directory of downloaded Stanford3dDataset_v1.2 dataset.
  # RAW_S3DIS_DIR: the directory of Stanford2d3dDataset_noXYZ dataset. (optional, for parsing normal)
  # PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset (output dir).
  
  # S3DIS without aligned angle
  python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR}
  # S3DIS with aligned angle
  python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --align_angle
  # S3DIS with normal vector (recommended, normal is helpful)
  python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --parse_normal
  python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --align_angle --parse_normal
  ```

- (Alternative) Our preprocess data can also be downloaded [[here](https://huggingface.co/datasets/Pointcept/s3dis-compressed
)] (with normal vector and aligned angle), please agree with the official license before downloading it.

- Link processed dataset to codebase.
  ```bash
  # PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
  mkdir data
  ln -s ${PROCESSED_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
  ```


## Quick Start

### Training
**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard, and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

For example:
```bash
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -p python -d scannet -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/scannet/semseg-pt-v2m2-0-base.py --options save_path=exp/scannet/semseg-pt-v2m2-0-base
```
**Resume training from checkpoint.** If the training process is interrupted by accident, the following script can resume training from a given checkpoint.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
# simply add "-r true"
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME} -r true
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${CHECKPOINT_PATH}
```
**Weights and Biases.**
Pointcept by default enables both `tensorboard` and `wandb`. There are some usage notes related to `wandb`:
1. Disable by set `enable_wandb=False`;
2. Sync with  `wandb` remote server by `wandb login` in the terminal or set `wandb_key=YOUR_WANDB_KEY` in config.
3. The project name is "Pointcept" by default, custom it to your research project name by setting `wandb_project=YOUR_PROJECT_NAME` (e.g. Sonata-Dev, PointTransformerV3-Dev)

### Testing
During training, model evaluation is performed on point clouds after grid sampling (voxelization), providing an initial assessment of model performance. ~~However, to obtain precise evaluation results, testing is **essential**~~ *(now we automatically run the testing process after training with the `PreciseEvaluation` hook)*. The testing process involves subsampling a dense point cloud into a sequence of voxelized point clouds, ensuring comprehensive coverage of all points. These sub-results are then predicted and collected to form a complete prediction of the entire point cloud. This approach yields  higher evaluation results compared to simply mapping/interpolating the prediction. In addition, our testing code supports TTA (test time augmentation) testing, which further enhances the stability of evaluation performance.

```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
# Direct
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```
For example:
```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d scannet -n semseg-pt-v2m2-0-base -w model_best
# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/scannet/semseg-pt-v2m2-0-base.py --options save_path=exp/scannet/semseg-pt-v2m2-0-base weight=exp/scannet/semseg-pt-v2m2-0-base/model/model_best.pth
```

## Train Instructions For Each Model

#### PTv3

[PTv3](https://arxiv.org/abs/2312.10035) is an efficient backbone model that achieves SOTA performances across indoor and outdoor scenarios. The full PTv3 relies on FlashAttention, while FlashAttention relies on CUDA 11.6 and above, make sure your local Pointcept environment satisfies the requirements.

If you can not upgrade your local environment to satisfy the requirements (CUDA >= 11.6), then you can disable FlashAttention by setting the model parameter `enable_flash` to `false` and reducing the `enc_patch_size` and `dec_patch_size` to a level (e.g. 128).

```bash
# Scratched S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base

# S3DIS 6-fold cross validation
# 1. The default configs are evaluated on Area_5, modify the "data.train.split", "data.val.split", and "data.test.split" to make the config evaluated on Area_1 ~ Area_6 respectively.
# 2. Train and evaluate the model on each split of areas and gather result files located in "exp/s3dis/EXP_NAME/result/Area_x.pth" in one single folder, noted as RECORD_FOLDER.
# 3. Run the following script to get S3DIS 6-fold cross validation performance:
export PYTHONPATH=./
python tools/test_s3dis_6fold.py --record_root ${RECORD_FOLDER}
```

#### PTv1

```bash
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v1-0-base -n semseg-pt-v1-0-base
```

#### OctFormer
OctFormer from _OctFormer: Octree-based Transformers for 3D Point Clouds_.
1. Additional requirements:
```bash
cd libs
git clone https://github.com/octree-nn/dwconv.git
pip install ./dwconv
pip install ocnn
```
2. Uncomment `# from .octformer import *` in `pointcept/models/__init__.py`.
2. Training with the following example scripts:
```bash
# ScanNet
sh scripts/train.sh -g 4 -d s3dis -c semseg-octformer-v1m1-0-base -n semseg-octformer-v1m1-0-base
```
