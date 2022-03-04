#!/bin/bash

##########################################################################################################
#                                                                                                        #
#  project-name: IIAI - Rodin - Audio                                                                    #
#  module-name: ITN                                                                                      #
#  authors: Karim Serkhane                                                                               #
#  copyright: G42                                                                                        #
#                                                                                                        #
#                	                Script to run live ITN                                            #
#                                                                                                        #
#                                                                                                        #
##########################################################################################################

# Require ITN docker image created with dbuild.sh script
#
# Usage:
# Run ITN docker container (once it was created using dbuild.sh script), run:
#
# 	./run_itn.sh language_code optional-compiler
#
# For instance:
#
#	./run_itn.sh fr-FR
#
# Or for instance: 
#
#	./run_itn.sh fr-FR --compiler

if [ "$#" -eq 3 ]; then
	if [ "$#" -eq 2 ]; then
		if [ "$#" -lt 1 ]; then
			echo -e "Missing language code"
			echo -e "For instance: ./run_itn.sh fr-FR"
			echo -e "Or for instance: ./run_itn.sh fr-FR --compiler"
			exit
		fi
		conda init bash && source /root/.bashrc && conda activate itn
		cd /app/fst-thrax-grammars-inv/$1/
		rules=`cat RULES`
		thraxrewrite-tester --far=itn.far --rules=$rules
	fi
fi
conda init bash && source /root/.bashrc && conda activate itn
cd /app/fst-thrax-grammars-inv/$1/
thraxmakedep itn.grm
make
rules=`cat RULES`

thraxrewrite-tester --far=itn.far --rules=$rules
