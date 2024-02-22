import copy
import datetime
import gc
import logging
import math
import os
import pickle
import random
import time

import numpy as np
import torch
from torch import nn

from util import postgres, envs, treeconv_dropout, DP
from util.encoding import TreeConvFeaturize
from util.train_test_splits import *
from util import pg_executor
from util.postgres import _run_explain
from util import plans_lib

from tqdm.auto import tqdm
import argparse
import wandb

# Import functions from train_stack script
from train_stack import *


# ==> getGMRL(testquery, model_levels, pg_latency_test, nodeFeaturizer, costCache, workload)
#
# testquery:        List of sql file names (without extension, i.e. ['1a', '1b'])
# model_levels:     Contains all dynamic programming models at model_levels[level]
# pg_latency_test:  Not needed, since here we're not intrested in relative comparisons
# nodeFeaturizer:   Featurizer to transform the queries (nodes)
# costCache:        Cache of costs, let's use what is already cached (though not added in github due to the size)
# workload:         Workload information

TIMEOUT_LIMIT = 3 * 60 * 1000


def main(args):
    wandb.init(mode='disabled')

    train_queries, test_queries = get_train_test_split(args.experiment)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    dp_models, _ = getModelsFromFile(args.max_level, args.model_path + '/BestTrainModel_' + args.logs_name + '_', device=DEVICE)

    workload = envs.STACK(envs.STACK.Params())
    workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(workload.workload_info.rel_names)

    nodeFeaturizer = plans_lib.PhysicalTreeNodeFeaturizer(workload.workload_info)

    # try to load cost cache
    cost_cache_file_path = os.path.join(args.model_path, 'cost_' + args.logs_name + '.pkl')
    if os.path.exists(cost_cache_file_path):
        with open(cost_cache_file_path, 'rb') as f:
            cost_cache = pickle.load(f)
            
        print(f"[Cost Cache]\tSuccessfully loaded cost cache from {cost_cache_file_path}")
    else:
        cost_cache = {}
        print(f"[Cost Cache]\tNo cost cache could be found at {cost_cache_file_path}")

    if os.path.exists(args.log_file_path):
        print(f"Logfile already exists at {args.log_file_path}")
        return
    else:
        # Write CSV header
        with open(args.log_file_path, 'w') as f:
            f.write(f"query_ident;split;inference_time;planning_time;execution_time")
            f.write(os.linesep)

    print('==='*30)
    print(f"\tTRAIN Queries ({len(train_queries)})")
    for train_query in train_queries:
        tmp = evaluate_query(train_query, dp_models, nodeFeaturizer, cost_cache, workload, train=True, log_file_path=args.log_file_path)
        output_string = ';'.join([str(x) for x in [tmp['query_ident'], tmp['split'], tmp['inference_time'], tmp['planning_time'], tmp['execution_time']]])
        write_to_logfile(args.log_file_path, output_string)
        
    print('==='*30)
    print(f"\tTEST Queries ({len(test_queries)})")
    for test_query in test_queries:
        tmp = evaluate_query(test_query, dp_models, nodeFeaturizer, cost_cache, workload, train=False, log_file_path=args.log_file_path)
        output_string = ';'.join([str(x) for x in [tmp['query_ident'], tmp['split'], tmp['inference_time'], tmp['planning_time'], tmp['execution_time']]])
        write_to_logfile(args.log_file_path, output_string)


def write_to_logfile(log_file_path, line):
    with open(log_file_path, 'a') as f:
        f.write(line)
        f.write(os.linesep)


def read_query(path):
    with open(path, 'r') as f:
        data = f.read().splitlines()
        # Remove lines that are just comments starting with "--"
        data = [line for line in data if not line.startswith('--')]
        return ' '.join(data)


def evaluate_query(query_ident, dp_models, nodeFeaturizer, cost_cache, workload, train, log_file_path):
    query_path = './../balsa/queries/stack/' + query_ident + '.sql'
    sql_query = read_query(query_path)

    inference_time_start = time.time()
    join_graph, all_join_conds, query_leaves, origin_dp_tables = DP.getPreCondition(query_path)

    selected_plan_hint, _ = DP.dp.TEST_left_prune_bayes(join_graph, all_join_conds, query_leaves, origin_dp_tables,
                                                        workload, dp_models, nodeFeaturizer, cost_cache)
    inference_time_end = time.time() - inference_time_start

    tmp = run_query(query_ident, sql_query, selected_plan_hint)
    tmp['inference_time'] = inference_time_end
    tmp['split'] = 'train' if train else 'test'

    print(f"[{query_ident}]\tInference [s]: {tmp['inference_time']:.4f}\tPlanning [ms]: {tmp['planning_time']:.4f}\tExecution [ms]: {tmp['execution_time']:.4f}")
    return tmp


def run_query(query_ident, sql_query, hint):
    with pg_executor.Cursor() as cursor:
        # GEQO must be disabled for hinting larger joins to work.
        # Why 'verbose': makes ParsePostgresPlanJson() able to access required
        # fields, e.g., 'Output' and 'Alias'.  Also see SqlToPlanNode() comment.
        try:
            result = _run_explain('explain(verbose, format json, analyze)',
                                sql_query,
                                hint,
                                verbose=False,
                                geqo_off=True,
                                cursor=cursor,
                                timeout_ms=TIMEOUT_LIMIT)
        except Exception as e:
            print(f"Query {query_ident} could not be executed:\n\t{e}")
            planning_time = 0
            execution_time = 2 * TIMEOUT_LIMIT

        else:
            result = result.result
            if result == []:
                print(f"An error occurred while executing Query {query_ident}.")
                planning_time = 0
                execution_time = 2 * TIMEOUT_LIMIT

            else:
                json_dict = result[0][0][0]
                
                planning_time = float(json_dict['Planning Time'])
                execution_time = float(json_dict['Execution Time'])

    return {
        'query_ident': query_ident,
        'planning_time': planning_time,
        'execution_time': execution_time
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='_0803-075110', help='Folder containing the training artefacts, i.e. cost cache, running log and model checkpoints')
    parser.add_argument('--max_level', type=int, default=12, help='maximum number of tables and aliases used in joins, for JOB this is 17 (template 29) and STACK is 12')
    parser.add_argument('--logs_name', type=str, default='', help='Name given to distinguish between experiments, variable "logs_name"')
    parser.add_argument('--log_file_path', type=str, required=True, help='Where the log of executions should be generated')
    parser.add_argument('--experiment', type=str, required=True, help='Defines which train-test split to use (see #get_train_test_split method).')
    
    args = parser.parse_args()
    main(args)

    # python3 test_job.py --log_file_path log__leave_one_out_1.csv --experiment LeaveOneOutSplit1 --model_path _0803-075110
