#!/bin/bash

##########################################################################################################
#                                                                                                        #
#  project-name: IIAI - Rodin - Audio                                                                    #
#  module-name: ITN                                                                                      #
#  authors: Karim Serkhane                                                                               #
#  copyright: G42                                                                                        #
#                                                                                                        #
#                	         Script to create the ITN docker Image                                    #
#                                                                                                        #
#                                                                                                        #
##########################################################################################################

# Requirements: 
#	- Docker: https://docs.docker.com/get-docker/
#	- Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
#	- Thrax Environment: conda install -c conda-forge thrax
#
# Usage:
# Open terminal, go inside of Thrax conda environment and run:
#
# 	./dbuild.sh
#
# Or you can also run:
# 
#	./dbuild.sh --conda
#

SCRIPT=$(readlink -f $0)
SCRIPT_DIR=`dirname $SCRIPT`

if [ $1 == "--conda" ]; then
	cd $SCRIPT_DIR
	source ~/anaconda3/etc/profile.d/conda.sh
	conda env create --file environment.yml
	conda init bash
	conda activate itn
fi

for dir in fst-thrax-grammars-inv/* ; do
	folder="$(basename -- $dir)"
	if [ $folder != "common" ]; then
		cd $SCRIPT_DIR/$dir
		thraxmakedep itn.grm
		make
	fi
done

cd $SCRIPT_DIR
docker build -t itn .

