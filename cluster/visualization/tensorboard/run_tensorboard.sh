#!/bin/bash

##########################################################################################################
#                                                                                                        #
#                                                                                                        #
#                Script to run locally tensorboard with results from a remote server                     #
#                                                                                                        #
#                                                                                                        #
##########################################################################################################

# Require Tensorboard, Firefox and SSHPASS installed on local


if [ "$#" -lt 3 ]; then
	if [ "$#" -lt 2 ]; then
		if [ "$#" -lt 1 ]; then
			echo -e "Missing parent output models directory which contains all experiments and cluster address and password."
			echo -e "For instance: ./run_tensorboard /home/ks1/models/tts/flowtron/ 'abc@140.30.20.10' 'password123'"
			exit
		fi
		echo -e "Missing cluster address and password."
		echo -e "For instance: ./run_tensorboard /home/ks1/models/tts/flowtron/ 'abc@140.30.20.10' 'password123'"
		exit
	fi
	echo -e "Missing cluster password"
	echo -e "For instance: ./run_tensorboard /home/ks1/models/tts/flowtron/ 'abc@140.30.20.10' 'password123'"
	exit
fi

if [ "$#" -lt 4 ]; then
	dir_home_cluster='/home/ks1'
else
	dir_home_cluster=$4
fi

if [ "$#" -eq 5 ]; then
	dir_env_tensorboard=$5
	source $dir_env_tensorboard
fi

dir_tensorboard_logs=$1
dir_tensorboard_logs=$(echo $dir_tensorboard_logs | sed 's/\/$//g')
dir_log=$(echo $dir_tensorboard_logs | cut -d "/" -f4-)
dir_log_local=$(echo $dir_log | rev | cut -d "/" -f2- | rev)

remote_address=$2
password_cluster=$3

mkdir -p $HOME/$dir_log_local
sshpass -p "$password_cluster" rsync -avr --exclude={'model_*','*.pt'} $remote_address:$dir_home_cluster/$dir_log $HOME/$dir_log_local

pkill -9 -x 'tensorboard'
tensorboard --logdir=$HOME/$dir_log --port 8080 & (sleep 2; firefox --new-window http://localhost:8080/)

