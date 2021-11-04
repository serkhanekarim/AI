#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

from modules.Global.variable import Var
from modules.Global.method import Method
from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter

from tqdm import tqdm
    
def main(args):
    '''
    Test text normalizer using the Flowtron one based on python
    '''
       
    path_test = args.path_test
    module_normalizer = args.module_normalizer
    function_mappings = Var().TEXT_NORMALIZATION_CLEANERS_FUNCTION_MAPPINGS
    normalizer = Method().call_func(function_mappings,module_normalizer)
    
    '''
    Get unit test from path_test and get failed test
    '''
    obj = {'header':None}
    unit_test = DataReader(path_file=path_test,filetype="sv",separator="\t").read_data_file(**obj)
    print(unit_test)
    sentences_normalized = unit_test[unit_test.columns[0]].apply(lambda sentence : normalizer(sentence))
    sentences_target = unit_test[unit_test.columns[1]]
    results_test = [(str(sentence_normalized) + "\t" + str(sentences_target[index]),print("FAIL TEST: " + str(sentence_normalized) + "\t" + str(sentences_target[index])))[0] 
                    for index,sentence_normalized in enumerate(tqdm(sentences_normalized)) if str(sentence_normalized) != str(sentences_target[index])]
    
    '''
    Write results test
    '''
    test_filename = filename = Method().get_filename(path_test)
    filename_results = "results" + "_" + test_filename + ".txt"
    path_results = os.path.join(directory_of_results,filename_results) 
    DataWriter(results_test, path_results).write_data_file()
    


if __name__ == "__main__":
    
    PROJECT_NAME = os.path.splitext(os.path.basename(__file__))[0]
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,"DATA",PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_test", help="Path of a file containing Youtube urls", required=True, nargs='?')
    parser.add_argument("-module_normalizer", help="Language to select for subtitle", required=True, nargs='?')
    
    
    args = parser.parse_args()
    
    main(args)    