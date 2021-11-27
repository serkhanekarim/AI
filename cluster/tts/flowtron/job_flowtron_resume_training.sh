#!/bin/bash
#SBATCH -n 80
#SBATCH --gres=gpu:v100:8
#SBATCH --time=48:00:00

path_config=$1
outdir=$2
checkpoint_name=$3
dir_flowtron='/home/ks1/Repositories/AI/modules/tts/flowtron/'
path_container='/home/ks1/containers/singularity/tts/flowtron/flowtron_p18.sif'

cd $dir_flowtron

module load singularity
singularity exec --nv $path_container python -m torch.distributed.launch --use_env --nproc_per_node=8 train.py -c $path_config -p train_config.checkpoint_path=$outdir/$checkpoint_name
