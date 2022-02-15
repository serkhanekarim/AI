#!/bin/bash

# This command will run the docker image named itn-grammar

SCRIPT=$(readlink -f $0)
SCRIPT_DIR=`dirname $SCRIPT`

docker run -it itn-grammars .
