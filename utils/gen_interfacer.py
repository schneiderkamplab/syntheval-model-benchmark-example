# Description: Script for generating synthetic datasets in a variety of ways
# Author: Anton D. Lautrup
# Date: 03-04-2024

import os
import subprocess

import pandas as pd

from .utils import dataset_train_test

from synthcity.plugins import Plugins

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

def _load_data(train_data_name):
    try:
        df_train = pd.read_csv(train_data_name +'_train.csv').dropna()
    except:
        dataset_train_test(train_data_name, 0.8)
        df_train = pd.read_csv(train_data_name +'_train.csv').dropna()
    return df_train

def _use_synthcity(train_data_name, gen_model):
    df_train = _load_data(train_data_name)

    syn_model = Plugins().get(gen_model, n_iter=2000)
    syn_model.fit(df_train)

    df_syn = syn_model.generate(count=len(df_train)).dataframe()

    return df_syn


def _use_synthpop(train_data_name, gen_model):
    df_train = _load_data(train_data_name)
    command = [
                "Rscript",
                "utils/synthpop.R",
                train_data_name +"_train.csv",
                train_data_name + "_temp"
            ]
    
    subprocess.run(command, check=True)

    df_syn = pd.read_csv(train_data_name + '_temp.csv')
    df_syn.columns = [col for col in df_train.columns]
    os.remove(train_data_name + '_temp.csv')
    return df_syn

def _use_datasynthesizer(train_data_name, gen_model):
    df_train = _load_data(train_data_name)

    input_data = train_data_name +'_train.csv'
    description_file = 'synthesis_info_' + train_data_name + "_" + gen_model + ".json"

    describer = DataDescriber(category_threshold=10)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data, 
                                                            epsilon=0, 
                                                            k=2,
                                                            attribute_to_is_categorical={})
    describer.save_dataset_description_to_file(description_file)

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(len(df_train), description_file)

    df_syn = generator.synthetic_dataset
    
    return df_syn

def generate_synthetic_data(train_data_name, gen_model):
    if gen_model in ['ctgan','adsgan', 'tvae', 'nflow']:
        df_syn = _use_synthcity(train_data_name, gen_model)
    elif gen_model == 'synthpop':
        df_syn = _use_synthpop(train_data_name, gen_model)
    elif gen_model == 'datasynthesizer':
        df_syn = _use_datasynthesizer(train_data_name, gen_model)
    elif gen_model == 'debug':
        df_syn = _load_data(train_data_name)
    else:
        raise NotImplementedError("The chosen generative model could not be run!")

    return df_syn