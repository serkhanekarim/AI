#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import csv

from text.cleaners import basic_cleaners
from text.cleaners import transliteration_cleaners
from text.cleaners import flowtron_cleaners
from text.cleaners import english_cleaners

from tqdm import tqdm

import sys
dir_modules = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..","..")
sys.path.append(dir_modules)
from modules.Global.variable import Var
from modules.Global.method import Method
from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter
    
def main(args):
    '''
    Test text normalizer using the Flowtron one based on python
    '''
    
    TEXT_NORMALIZATION_CLEANERS_FUNCTION_MAPPINGS = {
        'basic_cleaners': basic_cleaners,
        'transliteration_cleaners': transliteration_cleaners,
        'flowtron_cleaners': flowtron_cleaners,
        'english_cleaners': english_cleaners
    }
       
    path_test = args.path_test
    module_normalizer = args.module_normalizer
    function_mappings = TEXT_NORMALIZATION_CLEANERS_FUNCTION_MAPPINGS
    normalizer = Method().call_func(TEXT_NORMALIZATION_CLEANERS_FUNCTION_MAPPINGS,module_normalizer)
    
    '''
    Get unit test from path_test and get failed test
    '''
    obj = {'header':None, 'na_filter':False, 'quoting':csv.QUOTE_NONE}
    unit_test = DataReader(path_file=path_test,filetype="sv",separator="\t").read_data_file(**obj)
    sentences_normalized = unit_test[unit_test.columns[0]].apply(lambda sentence : normalizer(sentence))
    sentences_target = unit_test[unit_test.columns[1]]
    results_test = ["Original\tNormalized\tExpected Normalization"]
    results_test += [(str(unit_test[unit_test.columns[0]][index]) + "\t" + str(sentence_normalized) + "\t" + str(sentences_target[index]),
                     print("FAIL TEST: " + str(unit_test[unit_test.columns[0]][index]) + " - (Original)" + " ||| " + str(sentence_normalized) + " - (Normalized)" + " ||| " + str(sentences_target[index]) + " (Expected Normalization)"))[0] 
                    for index,sentence_normalized in enumerate(tqdm(sentences_normalized)) if str(sentence_normalized) != str(sentences_target[index])]
    
    if len(results_test)-1 == 0: print("ALL TEST(S) PASSED!")
    else: print(str(len(results_test)-1) + "/" + str(unit_test.shape[0]) + " TEST(S) FAILED!")
    
    '''
    Write results test
    '''
    test_filename = Method().get_filename(path_test)
    filename_results = "results" + "_" + test_filename + ".txt"
    path_results = os.path.join(directory_of_results,filename_results) 
    DataWriter(results_test, path_results).write_data_file()
    


if __name__ == "__main__":
    
    #./text-normalizer.py -path_test '/home/serkhane/Repositories/AI/modules/preprocessing/text_normalizer/test/unit_test-main-en.txt' -module_normalizer 'flowtron_cleaners'
    
    PROJECT_NAME = os.path.splitext(os.path.basename(__file__))[0]
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    #directory_of_data = os.path.join(directory_of_script,"DATA",PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    #os.makedirs(directory_of_data,exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-path_test", help="Path of a file containing Youtube urls", required=True, nargs='?')
    parser.add_argument("-module_normalizer", help="Language to select for subtitle", required=True, nargs='?')
    
    
    args = parser.parse_args()
    
    main(args)    