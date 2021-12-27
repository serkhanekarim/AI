#!/bin/bash
#SBATCH -n 80
#SBATCH --gres=gpu:v100:8
#SBATCH --time=48:00:00

##########################################################################################################
#                                                                                                        #
#                                                                                                        #
#                	      Script to run flowtron batch using cluster                                  #
#                                                                                                        #
#                                                                                                        #
##########################################################################################################

# Require cluster
# Usage:
# Open a terminal in the same directory as ./job_flowtron_training.sh and to start a new training run:
#
# ./job_flowtron_training.sh /path/to/config.json
#
# or to resume training:
#
# ./job_flowtron_training.sh /path/to/config.json /path/to/checkpoint/model

dir_flowtron='/home/ks1/Repositories/AI/modules/tts/flowtron/'
path_container='/home/ks1/containers/singularity/tts/flowtron/flowtron_p18.sif'

cd $dir_flowtron

if [ "$#" -lt 2 ]; then
	if [ "$#" -lt 1 ]; then
		echo -e "Missing path to the config.json and/or path to the checkpoint model"
		echo -e "For instance: ./job_flowtron_training.sh /path/to/config.json (to start new training)"
		echo -e "For instance: ./job_flowtron_training.sh /path/to/config.json /path/to/checkpoint/model (to resume training from a checkpoint)"
		exit
	fi
	path_config=$1
	module load singularity
	singularity exec --nv $path_container python -m torch.distributed.launch --use_env --nproc_per_node=8 train.py -c $path_config
fi

path_config=$1
path_checkpoint_name=$2
module load singularity
singularity exec --nv $path_container python -m torch.distributed.launch --use_env --nproc_per_node=8 train.py -c $path_config -p train_config.checkpoint_path=$path_checkpoint_name

