import psycopg2
import os
import sys
import random
from time import time, sleep
import datetime
import argparse
import glob

TIMEOUT_LIMIT = 3 * 60 * 1000
NUM_EXECUTIONS = 3


def current_timestamp_str():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

def pg_connection_string(db_name):
    return f"dbname={db_name} user=postgres password=postgres host=pg_bao"

def run_query(sql, bao_select=False, bao_reward=False, db_name='imdbload', use_geqo=True, use_bao=True):
    measurements = []
    try:
        conn = psycopg2.connect(pg_connection_string(db_name=db_name))
        cur = conn.cursor()
        # Hardcode bao_host to fixed IP given in docker-compose
        cur.execute("SET bao_host TO '10.5.0.6'")
        cur.execute(f"SET enable_bao TO {bao_select or bao_reward}")
        cur.execute(f"SET enable_bao_selection TO {bao_select}")
        cur.execute(f"SET enable_bao_rewards TO {bao_reward}")
        cur.execute("SET bao_num_arms TO 5")
        cur.execute(f"SET statement_timeout TO {TIMEOUT_LIMIT}")
        
        if not use_geqo:
            cur.execute(f"SET geqo TO off")

        for i in range(NUM_EXECUTIONS):
            cur.execute(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {sql}") 
            result = cur.fetchall()[0][0]
            # we are explicitly interested in execution *plus* planning time for testing

            bao_hint = result[0]['Bao']['Bao recommended hint'] if use_bao else None
            measurements.append({
                'execution_time': result[-1]['Execution Time'],
                'planning_time': result[-1]['Planning Time'],
                'hint': bao_hint
            })
            print(f"\t{i}: Execution Time: {measurements[-1]['execution_time']:.4f}\tPlanning Time: {measurements[-1]['planning_time']:.4f}")
            
        conn.close()
    except Exception as e:
        print("An unexpected exception OR timeout occured during database querying:", e)
        conn.close()
        
        tmp = []
        for _ in range(NUM_EXECUTIONS):
            tmp.append({
                'execution_time': 2 * TIMEOUT_LIMIT,
                'planning_time': 2 * TIMEOUT_LIMIT,
                'hint': None
            })
        return tmp

    return measurements


def main(args):
    # Look for .sql files
    pattern = os.path.join(args.query_dir, '**/*.sql')
    query_paths = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(query_paths)} queries in {args.query_dir} and its subdirectories.")

    queries = []
    for fp in query_paths:
        with open(fp) as f:
            query = f.read()
        queries.append((fp, query))

    use_bao = args.use_bao and (not args.use_postgres)
    print("Using Bao:", use_bao)

    use_geqo = args.use_geqo
    print("Using GEQO:", use_geqo)

    db_name = args.database_name
    print("Running against DB:", db_name)

    random.seed(42)

    print(f"Start running {len(queries)} queries for evaluation...")

    if os.path.exists(args.output_file):
        raise FileExistsError(f"The file {args.output_file} already exists, stopping.")
    
    for fp, q in queries:
        measurements = run_query(q, bao_select=use_bao, bao_reward=False, db_name=db_name, use_geqo=use_geqo, use_bao=use_bao)
        for i, measurement in enumerate(measurements):
            output_string = f"{'x' if measurement['hint'] is None else measurement['hint']}, {i}, {current_timestamp_str()}, {fp}, {measurement['planning_time']}, {measurement['execution_time']}, {'Bao' if use_bao else 'PG'}"
            print(output_string)            
            with open(args.output_file, 'a') as f:
                f.write(output_string)
                f.write(os.linesep)


# Example Call:
#
# python3 run_test_queries.py --use_postgres --database_name imdbload --query_dir queries/join-order-benchmark --output_file test__postgres__job.txt
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_bao', action='store_true', help='Should Bao be used?')
    parser.add_argument('--use_postgres', action='store_true', help='Should Postgres be used?')
    parser.add_argument('--database_name', type=str, default='imdbload', help='Database name to query against')
    parser.add_argument('--query_dir', type=str, required=True, help='Directory which contains all the queries to evaluate')
    parser.add_argument('--output_file', type=str, required=True, help='File in which to store the results')
    parser.add_argument('--use_geqo', action='store_true', default=True, help='Should GEQO be used? (default=True)')
    parser.add_argument('--disable_geqo', action='store_false', dest='use_geqo', help='Should GEQO be disabled?')

    args = parser.parse_args()

    if (not args.use_bao) and (not args.use_postgres):
        print(f"Need to either select Bao or Postgres to be used. (--use_bao or --use_postgres)")
    if args.use_bao and args.use_postgres:
        print("Need to only select Bao or Postgres, not both. (--use_bao or --use_postgres)")

    main(args)
