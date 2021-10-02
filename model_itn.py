#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import os
import wget
import pynini
import nemo_text_processing

from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
    
def main(args):

    inverse_normalizer = InverseNormalizer(lang='en')
    
    raw_text = "we paid one hundred and twenty three dollars for this desk, and this."
    inverse_normalizer.inverse_normalize(raw_text, verbose=False)

if __name__ == "__main__":
    
    PROJECT_NAME = "NeMo_ITN"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,"DATA",PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_directory", help="Directory of location of the data for training", required=False,  default=directory_of_data, nargs='?')

    
    args = parser.parse_args()
    
    main(args)    