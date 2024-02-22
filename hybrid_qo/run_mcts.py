import random
random.seed(113)

import sys
import wandb
from datetime import datetime
import json
from tqdm.auto import tqdm

from ImportantConfig import Config
from sql2fea import TreeBuilder,value_extractor
from NET import TreeNet
from sql2fea import Sql2Vec
from TreeLSTM import SPINN

from Hinter import Hinter
from mcts import MCTSHinterSearch


def load_queries(queries_path):
    with open(queries_path) as f:
        queries = json.load(f)
    return queries

def main(config):
    run_name = datetime.now().strftime('%Y_%m_%d__%H%M%S')
    wandb.init(
        project='hybrid_qo',
        entity='FILL_IN_YOUR_WANDB_ENTITY_HERE',
        name=run_name,
        config=config.__dict__
    )

    train_queries = load_queries(config.queries_file)
    test_queries = load_queries(config.queries_file.replace('__train', '__test'))

    tree_builder = TreeBuilder()
    sql2vec = Sql2Vec()
    #table_num = config.max_alias_num? or actually number of tables?
    value_network = SPINN(head_num=config.head_num, input_size=config.input_size, hidden_size=config.hidden_size, table_num = 50,sql_size = config.max_alias_num*config.max_alias_num+config.max_column).to(config.device)
    for name, param in value_network.named_parameters():
        from torch.nn import init
        if len(param.shape)==2:
            init.xavier_normal_(param)
        else:
            init.uniform_(param)

    net = TreeNet(tree_builder= tree_builder,value_network = value_network)
    mcts_searcher = MCTSHinterSearch()
    hinter = Hinter(model = net,sql2vec = sql2vec,value_extractor = value_extractor,mcts_searcher = mcts_searcher)

    print(len(train_queries))

    # Prepare query log file
    query_log_file_path = f"logs/{run_name}__query_log.csv"
    columns = ['epoch', 'test_query', 'query_ident', 'pg_plan_time', 'pg_latency', 'mcts_time', 'hinter_plan_time', 'MPHE_time', 'hinter_latency', 'hinter_query_ratio']
    with open(query_log_file_path, 'w') as f:
        f.write(','.join(columns) + '\n')

    # Since the splits provided by Lehmann, Sulimov & Stockinger do not include 20'000 queries,
    # we instead run the approx. 80-90 queries in repeated epochs to achieve a roughly similar
    # amount of executed queries, though these include the same queries multiple times!
    #
    for epoch in tqdm(range(config.n_epochs), total=config.n_epochs, desc='Iterating over epochs...'):
        train_epoch(hinter, train_queries, epoch, query_log_file_path)

        if epoch % 10 == 0:
            test_epoch(hinter, test_queries, epoch, query_log_file_path)
    
    # Final eval
    test_epoch(hinter, test_queries, epoch, query_log_file_path)


def train_epoch(hinter, queries, epoch, query_log_file_path):
    s_pg = 0
    s_hinter = 0

    # Because of multiple epochs, make sure that queries are randomly shuffled at each time
    random.shuffle(queries)

    pbar = tqdm(enumerate(queries[:]), total=len(queries), leave=False, desc='Iterating over training queries...')
    for idx, (sql, query_ident, _) in pbar:
        pbar.set_description(f"Iterating over training query {query_ident}...")
        
        pg_plan_time, pg_latency, mcts_time, hinter_plan_time, MPHE_time, hinter_latency, actual_plans, actual_time = hinter.hinterRun(sql)
        pg_latency /= 1000
        hinter_latency /= 1000
        pg_plan_time /= 1000
        hinter_plan_time /= 1000

        s_pg += pg_latency
        s_hinter += sum(actual_time) / 1000

        wandb.log({
            'epoch': epoch,
            'pg_plan_time': pg_plan_time,
            'pg_lateny': pg_latency,
            'mcts_time': mcts_time,
            'hinter_plan_time': hinter_plan_time,
            'MPHE_time': MPHE_time,
            'hinter_latency': hinter_latency,
            'hinter_global_ratio': s_hinter / s_pg,
            'hinter_query_ratio': pg_latency / (sum(actual_time) / 1000),
            'query_ident': query_ident,
            'test_query': 0
        })

        with open(query_log_file_path, 'a') as f:
            f.write(f"{epoch},0,{query_ident},{pg_plan_time},{pg_latency},{mcts_time},{hinter_plan_time},{MPHE_time},{hinter_latency},{pg_latency / (sum(actual_time) / 1000)}\n")


def test_epoch(hinter, queries, epoch, query_log_file_path):
    s_pg = 0
    s_hinter = 0

    pbar = tqdm(enumerate(queries[:]), total=len(queries), desc='Iterating over test queries...')
    for idx, (sql, query_ident, _) in pbar:
        pbar.set_description(f"Iterating over test query {query_ident}...")
        
        pg_plan_time, pg_latency, mcts_time, hinter_plan_time, MPHE_time, hinter_latency, actual_plans, actual_time = hinter.hinterRun(sql)
        pg_latency /= 1000
        hinter_latency /= 1000
        pg_plan_time /= 1000
        hinter_plan_time /= 1000

        s_pg += pg_latency
        s_hinter += sum(actual_time) / 1000

        wandb.log({
            'epoch': epoch,
            'pg_plan_time': pg_plan_time,
            'pg_lateny': pg_latency,
            'mcts_time': mcts_time,
            'hinter_plan_time': hinter_plan_time,
            'MPHE_time': MPHE_time,
            'hinter_latency': hinter_latency,
            'hinter_global_ratio': s_hinter/s_pg,
            'hinter_query_ratio': pg_latency / (sum(actual_time) / 1000),
            'query_ident': query_ident,
            'test_query': 1
        })

        with open(query_log_file_path, 'a') as f:
            f.write(f"{epoch},1,{query_ident},{pg_plan_time},{pg_latency},{mcts_time},{hinter_plan_time},{MPHE_time},{hinter_latency},{pg_latency / (sum(actual_time) / 1000)}\n")

if __name__ == '__main__':
    config = Config()
    main(config)
