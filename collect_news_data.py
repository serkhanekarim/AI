#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import os
import argparse
from modules.scraping.text import BBCScraper
from modules.reader.reader import DataReader
from modules.Global.method import Method

from tqdm import tqdm
from multiprocessing import Pool

def main(args, project_name):
    '''
    Scrap news data
    '''
    
    DATA_FOLDER_NAME = "DATA"
    
    data_directory = args["data_directory"]
    path_list_url = args["path_list_url"]
    
    directory_of_script = os.path.dirname(os.path.realpath(__file__))
    directory_of_results = os.path.join(directory_of_script,"results",PROJECT_NAME)
    directory_of_data = os.path.join(directory_of_script,DATA_FOLDER_NAME,PROJECT_NAME)
    os.makedirs(directory_of_results,exist_ok=True)
    os.makedirs(directory_of_data,exist_ok=True)
    if data_directory is None: data_directory = directory_of_data

    list_url = DataReader(path_list_url).read_data_file()
    
    [print(BBCScraper(url).get_title()) for url in list_url]
        


if __name__ == "__main__":
    
    PROJECT_NAME = "collect_news_data"
    
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