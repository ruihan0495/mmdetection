#!/bin/bash

#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:2
#SBATCH --job-name=ExampleJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=1:00:00
#SBATCH --mem=90G
#SBATCH --output=slurm_output_%A.out

module purge
module load 2019
module load Python/3.7.5-foss-2019b
module load CUDA/10.1.243
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module load Anaconda3/2018.12

source activate mmalb

cp -r $HOME/mmdetection/data "$TMPDIR"
cp -r $HOME/mmdetection/configs "$TMPDIR"
cd "$TMPDIR"
pwd
ls

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun python -u $HOME/mmdetection/tools/train.py $CONFIG --launcher="slurm" ${@:2}

cp -r work_dirs/cascade_df2 $HOME/mmdetection/scratch_output