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

# Require cluster and singularity
# Usage:
# Open a terminal in the same directory as ./job_flowtron_training.sh and to start a new training run:
#
# ./job_flowtron_training.sh /path/to/singularity_name.sif /path/to/config.json
#
# or to resume training:
#
# ./job_flowtron_training.sh /path/to/singularity_name.sif /path/to/config.json /path/to/checkpoint/model

dir_flowtron='/home/ks1/Repositories/AI/modules/tts/flowtron/'

cd $dir_flowtron

if [ "$#" -lt 4 ]; then
	if [ "$#" -lt 3 ]; then
		if [ "$#" -lt 2 ]; then
			echo -e "Missing path to the singularity.sif file and/or to the config.json and/or path to the checkpoint model"
			echo -e "For instance: ./job_flowtron_training.sh /path/to/singularity_name.sif /path/to/config.json (to start new training)"
			echo -e "For instance: ./job_flowtron_training.sh /path/to/singularity_name.sif /path/to/config.json /path/to/checkpoint/model (to resume training from a checkpoint)"
			exit
		fi
		path_container=$1
		path_config=$2
		module load singularity
		singularity exec --nv $path_container python -m torch.distributed.launch --use_env --nproc_per_node=8 train.py -c $path_config
	fi
	path_container=$1
	path_config=$2
	path_checkpoint_name=$3
	module load singularity
	singularity exec --nv $path_container python -m torch.distributed.launch --use_env --nproc_per_node=8 train.py -c $path_config -p train_config.checkpoint_path=$path_checkpoint_name
fi
