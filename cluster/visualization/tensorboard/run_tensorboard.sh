#!/bin/bash

##########################################################################################################
#                                                                                                        #
#                                                                                                        #
#                Script to run locally tensorboard with results from a remote server                     #
#                                                                                                        #
#                                                                                                        #
##########################################################################################################

# Require Tensorboard, Firefox and SSHPASS installed on local
# Usage:
# Open a terminal in the same directory as ./run_tensorboard.sh and run:
#
# ./run_tensorboard.sh /home/ks1/models/tts/flowtron/ 'abc@140.30.20.10' 'password123'
#


if [ "$#" -lt 3 ]; then
	if [ "$#" -lt 2 ]; then
		if [ "$#" -lt 1 ]; then
			echo -e "Missing parent output models directory which contains all experiments and cluster address and password."
			echo -e "For instance: ./run_tensorboard.sh /home/ks1/models/tts/flowtron/ 'abc@140.30.20.10' 'password123'"
			exit
		fi
		echo -e "Missing cluster address and password."
		echo -e "For instance: ./run_tensorboard.sh /home/ks1/models/tts/flowtron/ 'abc@140.30.20.10' 'password123'"
		exit
	fi
	echo -e "Missing cluster password"
	echo -e "For instance: ./run_tensorboard.sh /home/ks1/models/tts/flowtron/ 'abc@140.30.20.10' 'password123'"
	exit
fi

dir_tensorboard_logs=$1
dir_tensorboard_logs=$(echo $dir_tensorboard_logs | sed 's/\/$//g')
dir_log=$(echo $dir_tensorboard_logs | cut -d "/" -f4-)
dir_log_local=$(echo $dir_log | rev | cut -d "/" -f2- | rev)

remote_address=$2
password_cluster=$3

mkdir -p $HOME/$dir_log_local
sshpass -p "$password_cluster" rsync -avr --exclude={'model_*','*.pt'} $remote_address:$dir_tensorboard_logs $HOME/$dir_log_local

pkill -9 -x 'tensorboard'
tensorboard --logdir=$HOME/$dir_log --port 8080 & (sleep 5; firefox --new-window http://localhost:8080/)

