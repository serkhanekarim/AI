#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import os
import argparse
import math
import numpy as np
import timeit

from modules.reader.reader import DataReader
from modules.writer.writer import DataWriter
from modules.Global.method import Method

from modules.preprocessing.text.cleaning.make_lm_text import process_line

from tqdm import tqdm
from multiprocessing import Pool

import kenlm

def main(args, project_name):
    '''
    Prepare language model using KenLM
    '''
    
    DATA_FOLDER_NAME = "DATA"
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    
    language = args["language"].lower()
    clean_data = args["clean_data"]
    data_directory = args["data_directory"]
    path_list_training_data = args["path_list_training_data"]
    path_validation_data = args["path_validation_data"]
    path_list_training_data_cleaned = args["path_list_training_data_cleaned"]
    path_language_model = args["path_language_model"]
    perplexity = args["perplexity"]
    
    dir_text = os.path.join(directory_of_results,language)
    os.makedirs(dir_text,exist_ok=True)
    if data_directory is None: 
        data_directory = os.path.join(directory_of_script,DATA_FOLDER_NAME,PROJECT_NAME)
        os.makedirs(data_directory,exist_ok=True)
    
    list_path_training_text = DataReader(path_list_training_data).read_data_file(keep_line_break=False)
    if path_list_training_data_cleaned is None:
        path_list_training_data_cleaned = os.path.join(directory_of_results,"list_training_data_cleaned.txt")
        list_path_training_text_cleaned = []
    else:
        list_path_training_text_cleaned = DataReader(path_list_training_data_cleaned).read_data_file(keep_line_break=False)
    
    if clean_data:
        print("Cleaning data...")
        for path_text in tqdm(list_path_training_text):
            text = DataReader(path_text).read_data_file(keep_line_break=False)
            if len(text) > 0:
                #Fix number of max parallelized process
                nb_max_parallelized_process = min(len(text), os.cpu_count())
                if language == "ar":
                    print("ArbTextProcessor...")
                    from modules.preprocessing.text.cleaning.arb_text_processor import ArbTextProcessor
                    with Pool(processes=nb_max_parallelized_process) as pool:
                        text = pool.map(ArbTextProcessor().clean, tqdm(set(text)))
                    # list_arg = [(line, 4, 'english') for line in text]
                print("Process_line...")
                with Pool(processes=nb_max_parallelized_process) as pool:
                    text = pool.map(process_line, tqdm(set(text)))
                text = [element[0] for element in tqdm(text) if len(element) > 0]
                
                original_filename = Method().get_filename(path_text)
                cleaned_filename = original_filename + "_" + "cleaned" + "_" + language + ".txt"
                path_filename_cleaned = os.path.join(dir_text,cleaned_filename)
                list_path_training_text_cleaned.append(path_filename_cleaned)
                DataWriter(text, path_filename_cleaned).write_data_file()
        DataWriter(set(list_path_training_text_cleaned), path_list_training_data_cleaned).write_data_file()
    
    print("Concatening cleaned data...")
    total_filename = "total_cleaned_text.txt"
    path_total_filename = os.path.join(dir_text,total_filename)
    os.system("rm -rf " + path_total_filename)
    (os.system("cat " + path_text_cleaned + " >> " + path_total_filename) for path_text_cleaned in tqdm(list_path_training_text_cleaned))
    print("Removing duplicated line...")
    os.system("sort " + path_total_filename + " | uniq -u > " + path_total_filename)
    
    if perplexity:
        print("Compute LM perplexity of validation data...")
        validation_text = DataReader(path_validation_data).read_data_file(keep_line_break=False)
        validation_filename = Method().get_filename(path_validation_data)
        perplexity_filename = validation_filename + "_" + "perplexity" + ".txt"
        path_perplexity_filename = os.path.join(dir_text,perplexity_filename)
        model = kenlm.Model(path_language_model)        
        perplexity_data = [s + "\t" + str(model.perplexity(s)) + "\t" + str(math.pow(np.prod([math.pow(10.0, score) for score, _, _ in model.full_scores(s)]), 1.0/len(list(model.full_scores(s))))) for s in validation_text]
        DataWriter(perplexity_data, path_perplexity_filename).write_data_file()
        

if __name__ == "__main__":
    
    PROJECT_NAME = "prepare_language_model"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='JSON file for configuration')
    parser.add_argument('-p', '--params', nargs='+', default=[])
    args = parser.parse_args()
    args.rank = 0
    
    with open(args.config) as f:
        data = f.read()
    
    args_config = json.loads(data)[PROJECT_NAME]
    args_config = Method().update_params(args_config, args.params)
    
    main(args=args_config, project_name=PROJECT_NAME)