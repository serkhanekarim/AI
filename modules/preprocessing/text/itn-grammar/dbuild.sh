#!/bin/bash

# This command will build docker image named itn-grammar

SCRIPT=$(readlink -f $0)
SCRIPT_DIR=`dirname $SCRIPT`

cd $SCRIPT_DIR

docker build -t itn-grammars .
