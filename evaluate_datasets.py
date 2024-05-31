import glob, os

import numpy as np
import pandas as pd
import multiprocessing as mp

from syntheval import SynthEval
from utils.utils import results_formatting, rank_results_formatting

np.random.seed(0)   # remove seed lock for true randomness

### File stuff
paths_and_targets = {
    'data/small_few_atts/diabetes': 'Outcome',
    'data/small_few_atts/penguins': 'species',
    'data/small_few_atts/titanic': 'Survived',
    'data/small_some_atts/cervical_cancer': 'Biopsy',
    'data/small_some_atts/derm': 'class',
    'data/small_some_atts/spect': 'OVERALL_DIAGNOSIS',
    'data/small_many_atts/diabetic_mellitus': 'TYPE',
    'data/small_many_atts/mice_protein': 'class',
    'data/small_many_atts/spectrometer': 'ID-type',
    'data/large_few_atts/space_titanic': 'Transported',
    'data/large_few_atts/stroke': 'stroke',
    'data/large_few_atts/winequality': 'quality',
    'data/large_some_atts/cardiotocography': 'Class',
    'data/large_some_atts/one_hundred_plants': 'Class',
    'data/large_some_atts/steel_faults': 'Class',
    'data/large_many_atts/bankruptcy': 'Bankrupt',
    'data/large_many_atts/speed_dating': 'match',
    'data/large_many_atts/yeast_ml8': 'class1',
}

generative_model_names = ['ctgan', 'datasynthesizer', 'synthpop']

metrics = {
    "cio"       : {"confidence": 95},
    "corr_diff" : {"mixed_corr": True},
    "mi_diff"   : {},
    "ks_test"   : {"sig_lvl": 0.05, "n_perms": 1000},
    "h_dist"    : {},
    "p_mse"     : {"k_folds": 5, "max_iter": 100, "solver": "liblinear"},
    "nnaa"      : {"n_resample": 30},
    "cls_acc"   : {"F1_type": "micro", "k_folds": 5},
    "hit_rate"  : {"thres_percent": 0.0333},
    "eps_risk"  : {},
    "mia_risk"  : {"num_eval_iter": 5}
}


def process_data(paths_and_targets):
    data_path, target = paths_and_targets

    df_train = pd.read_csv(data_path + '_train.csv')
    df_test = pd.read_csv(data_path + '_test.csv')

    df_syn_dict = {model: pd.read_csv(data_path + '_'+ model +'.csv') for model in generative_model_names}

    SE = SynthEval(df_train, df_test, verbose=False)
    bmk_df, rnk_df = SE.benchmark(df_syn_dict, target, rank_strategy='summation', **metrics)

    ### Format and put results into the global results dataframe
    bmk_df = results_formatting(bmk_df, data_path)
    rnk_df = rank_results_formatting(rnk_df, data_path)

    return bmk_df, rnk_df


if __name__ == '__main__':
    ### Create a pool of workers
    with mp.Pool(mp.cpu_count()-2) as pool:
        results = pool.map(process_data, paths_and_targets.items())
    bmk_df, rnk_df = zip(*results)
    
    global_bmk_res = pd.concat(bmk_df, axis=0).reset_index(drop=True)
    global_rnk_res = pd.concat(rnk_df, axis=0).reset_index(drop=True)

    ### Cleanup
    for f in glob.glob("SE_benchmark_*.csv"):
        os.remove(f)

    ### Save results
    global_bmk_res.to_csv('evaluation_results.csv', index=False)
    global_rnk_res.to_csv('rank_results.csv', index=False)