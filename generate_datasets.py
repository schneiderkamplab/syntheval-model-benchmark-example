# Description: Script for creating synthetic data files used in the benchmark
# Author: Anton D. Lautrup
# Date: 07-05-2024

import glob, os

import numpy as np
import pandas as pd

from tqdm import tqdm
from time import process_time
from itertools import product
from joblib import Parallel, delayed

from syntheval import SynthEval
from utils.utils import prepare_directories
from utils.gen_interfacer import generate_synthetic_data

np.random.seed(0)   # remove seed lock for true randomness

### Parameters

NUM_REPEATS = 3

target_vars = {
    'data/small_few_atts/diabetes': 'Outcome',
    'data/small_few_atts/penguins': 'species',
    'data/small_few_atts/titanic': 'Survived',
    'data/small_some_atts/cervical_cancer': 'Biopsy',
    'data/small_some_atts/derm': 'class',
    'data/small_some_atts/spect': 'OVERALL_DIAGNOSIS',
    'data/small_many_atts/diabetic_mellitus': 'TYPE',
    'data/small_many_atts/mice_protein': 'class',
    'data/small_many_atts/spectrometer': 'ID-type',
    'data/large_few_atts/space_titanic': 'Survived',
    'data/large_few_atts/stroke': 'stroke',
    'data/large_few_atts/winequality': 'quality',
    'data/large_some_atts/cardiotocography': 'Class',
    'data/large_some_atts/one_hundred_plants': 'Class',
    'data/large_some_atts/steel_faults': 'class',
    'data/large_many_atts/bankruptcy': 'Bankrupt',
    'data/large_many_atts/speed_dating': 'match',
    'data/large_many_atts/yeast_ml8': 'class1'
}

generative_model_names = ['ctgan', 'datasynthesizer', 'synthpop']

GENERATE_MISSING = True
GENERATE_OVERWRITE = False

### Code to generate synthetic datasets
def work_func(iterable, num_reps = 3):
    dataset_name, generative_model =  iterable
    try: 
        if GENERATE_OVERWRITE: raise Exception("Regenerate the synthetic datasets")
        df_syn = pd.read_csv(dataset_name + '_' + generative_model + '.csv')
    except:
        if not GENERATE_MISSING: return None
        start_time = process_time()
        dfs = {i: generate_synthetic_data(dataset_name, generative_model) for i in range(num_reps)}
        
        end_time = process_time()
        elapsed_time = end_time - start_time
        
        df_train = pd.read_csv(dataset_name +'_train.csv')
        df_test = pd.read_csv(dataset_name + '_test.csv')

        SE = SynthEval(df_train, df_test, verbose=False)
        df_comb , _ = SE.benchmark(dfs, target_vars[dataset_name],'./fast_eval.json')

        idx = df_comb['rank'].idxmax() # get the best dataset

        dfs[idx].to_csv(dataset_name + '_' + generative_model + '.csv', index=False)

        for f in glob.glob("SE_benchmark_*.csv"):
            os.remove(f)

        return {"dataset": dataset_name, "model": generative_model, "time": elapsed_time/NUM_REPEATS}
    return None

prepare_directories(target_vars)

iterables_list = list(product(target_vars, generative_model_names))
elapsed_times = Parallel(n_jobs=-2)(delayed(work_func)(iterable, NUM_REPEATS) for iterable in tqdm(iterables_list))

# Create dataframe
df_res = pd.DataFrame([d for d in elapsed_times if d is not None])
df_res.to_csv('elapsed_times.csv', index=False)