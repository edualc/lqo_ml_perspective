# Installing Postgres, Balsa(+Neo), LEON & HybridQO

In this directory you find all the relevant files for running Postgres + [Balsa](https://github.com/balsa-project/balsa), Neo (through Balsa), [LEON](https://github.com/haitianchen/LEON) and [HybridQO](https://github.com/yxfish13/HyperQO/tree/main) in a docker environment.

>Marcus, Ryan, et al. "**Neo: A Learned Query Optimizer.**" Proceedings of the VLDB Endowment 12.11.

>Yang, Zongheng, et al. "**Balsa: Learning a Query Optimizer Without Expert Demonstrations.**" Proceedings of the 2022 International Conference on Management of Data. 2022.

>Chen, Xu, et al. "**LEON: a new framework for ml-aided query optimization.**" Proceedings of the VLDB Endowment 16.9 (2023): 2261-2273.

>Yu, Xiang, et al. "**Cost-based or learning-based? A hybrid query optimizer for query plan selection. (HybridQO)**" Proceedings of the VLDB Endowment 15.13 (2022): 3924-3936.

### Installation

The JOB database files are expected to be located in the `<repo>/data/` directory, outside the `<repo>/docker/` directory (see [Readme](../../data/README.md)). It contains the basic steps for downloading and preparing the JOB and STACK data (among other unused databases/workloads) for insertion into Postgres. The steps below assume, you already downloaded the JOB and STACK database and performed the preprocessing steps.

**This container was originally based to only run Balsa, but also includes running Neo (through different experiment configurations in Balsa), LEON and HybridQO (both as seperate folders mounted in the same container).**


### Building the Postgres + Balsa Docker containers

In order to differentiate between the container running Postgres and the container where Balsa and its code are run from, we call the Postgres container `pg_balsa` and the Balsa container `balsa` with an added `_img`-suffix for the images, i.e. `pg_balsa_img` and `balsa_img`. The Balsa container also mounts the codebase of LEON.


The following script `generate_docker_containers.sh` is available to build the containers:
```bash
cd /<repo>/docker/postgres_balsa
. generate_docker_containers.sh
```

**Note: We are basing our Dockerfile `Dockerfile__balsa_img` based on a NVIDIA CUDA base container which corresponds to the CUDA version used on our machine, version `11.8` (and later `12.2`). However, with a different CUDA version on your machines, it might be necessary to adjust the base container to _your_ CUDA version. You can find the different base images [here](https://hub.docker.com/r/nvidia/cuda), make sure to use the `ubuntu` version.**


#### Initializing the Postgres container

We assume the `imdbload` database is created by setting the `POSTGRES_DB` environment parameter through [docker-compose.yml](docker-compose.yml). The process of loading JOB into Postgres is taken from [Balsa](https://github.com/balsa-project/balsa) repository, see the [data/README.md](../../data/README.md) for more details. The following block assumes, you followed the steps outlined in the `data/README.md` directory, downloading IMDB and preprending the headers. After doing this, your `data/imdb` directory should contain the various `.csv` files of the IMDB tables.

We currently include instructions to load the `Join-Order-Becnhmark (JOB)` and `STACK` (introduced in [Bao](https://rmarcus.info/stack.html)).


```bash
# Start the container(s)
docker-compose up -d

# Open a different shell, since the one running docker-compose shows you the log output. If you want to run the container detached, add -d or --detach after docker-compose up, i.e. docker-compose up -d
#
# Start interactive shell in PG container
docker exec -it pg_balsa bash

# Change user to Postgres (linux user password is set to "postgres" in Dockerfile)
su postgres

cp /app/postgresql.conf /pgdata/postgresql.conf

# Restart Postgres instance (the container will stop)
pg_ctl restart

# Restart the container by stopping the docker-compose command in your other shell
docker-compose up -d
docker exec -it pg_balsa bash
su postgres

# LOAD JOB INTO DB
# ==========================================
#
# Run script to load the CSVs into the database. This process takes on the order of 15 minutes.
bash data/imdb/load-postgres/load_job_postgres.sh /data/imdb

# LOAD STACK INTO DB (Takes ~2.5h to restore dump, 17.4s to ANALYZE)
# ==========================================
#
psql -c "CREATE DATABASE so;"
pg_restore -d so -j 4 --clean --no-privileges --no-owner --verbose data/stack/so_pg12
psql -d so -c "ANALYZE verbose;"
```

Note, that Balsa adds [two additional indexes]((https://github.com/balsa-project/balsa/blob/main/load-postgres/fkindexes.sql)) to IMDB/JOB compared to the ones provided by [Leis et al.](https://github.com/gregrahn/join-order-benchmark/blob/master/fkindexes.sql):

```sql
create index subject_id_complete_cast on complete_cast(subject_id);
create index status_id_complete_cast on complete_cast(status_id);
```

##### Checking if pg_hint_plan is working

```bash
# Go into Postgres docker container
docker exec -it pg_balsa bash
su postgres

# Connect to imdbload database
psql -d imdbload

# Run the following queries to see, that pg_hint_plan is forcing the joins as expected
/*+ MergeJoin(ml lt) */ EXPLAIN ANALYZE SELECT COUNT(*) FROM movie_link AS ml JOIN link_type AS lt ON ml.link_type_id = lt.id;

/*+ HashJoin(ml lt) */ EXPLAIN ANALYZE SELECT COUNT(*) FROM movie_link AS ml JOIN link_type AS lt ON ml.link_type_id = lt.id;
```

The output should look something like this (using a MergeJoin between `ml` and `lt`):

```sql
 Aggregate  (cost=1886.88..1886.89 rows=1 width=8) (actual time=20.110..20.111 rows=1 loops=1)
   ->  Merge Join  (cost=0.42..1811.89 rows=29997 width=0) (actual time=0.049..17.399 rows=29997 loops=1)
         Merge Cond: (ml.link_type_id = lt.id)
         ->  Index Only Scan using link_type_id_movie_link on movie_link ml  (cost=0.29..1425.16 rows=29997 width=4) (actual time=0.039..11.435 rows=29997 loops=1)
               Heap Fetches: 29997
         ->  Index Only Scan using link_type_pkey on link_type lt  (cost=0.14..12.41 rows=18 width=4) (actual time=0.005..0.014 rows=17 loops=1)
               Heap Fetches: 17
 Planning Time: 0.433 ms
 Execution Time: 20.165 ms
(9 rows)
```


#### Initializing the Balsa container

```bash
# Start the container(s), if you have not already done so in the previous step
docker-compose up

# Start interactive shell in the Balsa container
docker exec -it balsa bash

cd /app/balsa

# Initialize the Conda environment
conda create -n balsa python=3.7 -y
conda activate balsa

# Install the necessary libraries
pip install -r requirements.txt
pip install -e .
pip install -e pg_executor

# Start Ray
# (You might get a warning saying you should set the environment variable as follows. This has no impact on execution times.)
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# Note: You might also get the message that Redis could not be started. In this case, Ray is likely already running. Check
# the log file referenced, such as at /tmp/ray/session_2023-10-31_15-09-08_244588_406/logs/redis.out
ray start --head


# Try the baseline, to see if Balsa is working (i.e. DB connection is available)
# WANDB_MODE=disabled makes it such that nothing is logged into Weights and Biases (WandB)
export WANDB_MODE=disabled; python run.py --run Baseline --local

# You can decide if you want to log experiment information into Weights and Biases (WandB), by setting
# either 'online' or 'disabled' into the `WANDB_MODE` environment variable, such as:
export WANDB_MODE=online
export WANDB_MODE=disabled



```


##### Ray

Originally, we had trouble running ray inside the Alpine distribution, but changing over to Debian fixed it. You can check that ```ray``` works, by running ```ray start --head```. This starts a Ray process that - as part of it - ensures that Redis is running. 

In version 1.9.2, there is a bug that obfuscated the error message of Redis not starting/being available by an AssertionError thrown in the `_initialize_internal_kv` method, but the real error is caused by `get_gcs_client` [here](https://github.com/ray-project/ray/blob/ef593fe5d3c864836b80ae77be32635cef42b537/python/ray/node.py#L453). Running on Ubuntu instead of Alpine (as the base Docker image) solved this.


##### Redis

You can check that Redis is running by calling ```redis-cli``` (```exit``` to close the Redis shell).


##### `protobuf` Error

Because we were running into the following error, we added protobuf=3.20.* to the `requirements.txt` file of the Balsa codebase.

```bash
TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower).
```

## Running Balsa Models

Balsa is based on experiment classes defined in the `balsa/experiments.py` file. Look for example at the definition of the `Balsa_JOBRandomSplit1` experiment (and its parent classes) to see example of its usage. The name of these classes is used in the `--run` parameter of the `balsa/run.py` script. Also add the `--local` flag, since we do not take advantage of the parallelized execution on multiple PostgreSQL instances at the same time. Experiments are automatically logged into [Weights and Biases](https://wandb.ai/site). You will be asked about your API code when first running an experiment with WandB - either sign up and create an account or export the environment variable `WANDB_MODE=disabled` to ignore any logging into WandB.

Generally, all JOB queries are taken as the dataset and you define the train/test split by setting different values for `test_query_glob` in your experiment config of the `balsa/experiments.py` file.

> **WARNING**: Make sure, that you change the `LOCAL_DSN` constant in the `balsa/pg_executor/pg_executor/pg_executor.py` file depending on the database that you want to run against. There are both versions for STACK and IMDB/JOB available.

### Running the Baseline

**Note:** Balsa requires, that the baseline has been run at least once to completion.

```bash
docker exec -it balsa bash

cd /app/balsa
conda activate balsa

# Try the baseline, to see if Balsa is working
# (i.e. DB connection is available) or any other experiment
export WANDB_MODE=run; python3 run.py --run Baseline --local
```

### Training a Balsa Model

> **WARNING**: The Balsa codebase by default assumes that you are running against JOB. In case you want to run STACK, change the workload in the `prepare` method of `balsa/balsa/experience.py` around line 200.

```bash
docker exec -it balsa bash

cd /app/balsa
conda activate balsa

python3 run.py --run Balsa_JOBRandomSplit1 --local
```

### Evaluating a Balsa Model

For the evaluation, you will find a file in the pattern `<timestamp>__<experiment_name>__plan_and_execute.txt`, which contains the results from your evaluation. The model checkpoint is typically found in the `balsa/wandb` folder, however we also added logging of checkpoints into the `balsa/checkpoints/<timestamp>/` folders to get additional model checkpoints beyond the most recent one.

```bash
docker exec -it balsa bash

cd /app/balsa
conda activate balsa

python3 test_model.py --run Balsa_JOBRandomSplit1 --model_checkpoint /app/balsa/wandb/run-20230713_142032-t6uvrt8o/files/checkpoint.pt
```

### General Notes about including Bitmap Scans for Balsa

- We have changed the Balsa code base to include Bitmap Index Scan, Bitmap Heap Scan and Tid Scan operators, as there was not given an explanation why the have been disabled in the configuration. We believe they were disabled, as the inclusion of Bitmap scans complicates the general parsing and makes the executed EXPLAIN ANALYZE sometimes differ from the hint that was passed (as i.e. HashJoin is replaced with NestLoop when a bitmap scan is one its children).
- Because of the previous point, we have disabled the hint checking mechanism, as we fully *expect* the executed_hint_str to be different from the hint_str in cases where Bitmap scans are present. You can re-enable the checks in the file `balsa/run.py` around line 257.
- Allowing the Balsa code base to work with Bitmap scans was done by:
  - Changing the `balsa/balsa/util/plans_lib.py` `plans_lib.Node` `#hint_str` method to change `BitmapHeapScan` and `BitmapIndexScan` into `BitmapScan`, to allow pg_hint_plan to use it.
  - Adding `Bitmap Heap Scan`, `Tid Scan` to both `balsa/balsa/envs/envs.py` and `balsa/experiments.py` to the default definition of scan operations (`Bitmap Index Scan` is not explicitly added as well, since BitmapIndexScan + BitmapHeapScan occur in tandem).
  - Enabling Tid and Bitmap Scans in the PostgreSQL configuration file
  - Treating the `Bitmap Heap Scan` node as the leaf node in `balsa/balsa/util/postgres.py` `#_parse_pg` method.
  - Treating the `Bitmap Heap Scan` node as the leaf node in `balsa/balsa/util/plans_lib.py` `GetAllSubtreesNoLeaves` `#_fn` method.
  - Treating the `Bitmap Heap Scan` node as the leaf node in `balsa/balsa/util/plans_lib.py` `GetAllSubtrees` `#_fn` method.
  - Treating the `Bitmap Heap Scan` node as the leaf node in `balsa/balsa/models/treeconv.py` `#_make_preorder_ids_tree` method.

#### Required changes in Balsa to add STACK as a new dataset

- Additional Experiment Configs in `balsa/experiments.py`
- New Workload Env in `balsa/balsa/envs/envs.py` as `STACK`, analogous to `JoinOrderBenchmark`
- Change to differentiate between Workloads in `_MakeWorkload` method of `balsa/run.py`
- Change in `RunBaseline` method to save initial policy under a different name
- Added `STACK_JOIN_GRAPH` in `balsa/balsa/util/graphs.py` and changed `balsa/balsa/experience.py` `prepare` method to distinguish between both graphs
- Added cached cardinalities to `GetAllTableNumRows` in `balsa/balsa/util/postgres.py`
- Changed `nx.Graph` to `nx.MultiGraph` in `_GetGraph` method of `balsa/balsa/util/simple_sql_parser.py`, since STACK contains queries that define relations between the same tables across multiple column pairs, i.e. `q1.owner_user_id = u1.id and q1.site_id = u1.site_id`
- Comment out assertion in `balsa/train_utils/train_utils.py` `Start` method of `Timer` class, since it fails when query executions time out.
- Adjusted cached pickle file paths in `Sim` class of `balsa/sim.py` to save separately for STACK 
- Treat all `psycopg2.OperationalError` in the `Execute` function in `pg_executor.py` as timeouts and disable the `wait_select_inter` callback.
- Replaced assertion in `SimpleQueryFeaturizer` (`balsa/sim.py`) with clamping for apparent cases where `est_rows > total_rows` OR `est_rows < 0`


## Running Neo Models

Running Neo models works the exact same was as Balsa (see above). However, the experiment name changes (as it requires a different set of configuration parameters). You can take a look at the `balsa/experiments.py` file and look for example at the definition of the `Neo_JOBRandomSplit1` experiment (and its parent classes) to see example of its usage.

> **WARNING**: Make sure, that you change the `LOCAL_DSN` constant in the `balsa/pg_executor/pg_executor/pg_executor.py` file depending on the database that you want to run against. There are both versions for STACK and IMDB/JOB available.

We have added a global timeout in particular for Neo

### Training a Neo Model

> **WARNING**: The Balsa codebase by default assumes that you are running against JOB. In case you want to run STACK, change the workload in the `prepare` method of `balsa/balsa/experience.py` around line 200.

When running Neo on JOB, make sure that you disable the timeout in the `Execute` method of the `balsa/pg_executor/pg_executor/pg_executor.py` file, such that Neo truly runs without a timeout again. We have currently set a global timeout of 5 minutes, such that Neo can actually progress.

```bash
docker exec -it balsa bash

cd /app/balsa
conda activate balsa

python3 run.py --run Neo_JOBRandomSplit1 --local
```

### Evaluating a Neo Model

The model checkpoint is typically found in the `balsa/wandb` folder, however we also added logging of checkpoints into the `balsa/checkpoints/<timestamp>/` folders to get additional model checkpoints beyond the most recent one.

```bash
docker exec -it balsa bash

cd /app/balsa
conda activate balsa

python3 test_model.py --run Neo_JOBRandomSplit1 --model_checkpoint /app/balsa/wandb/run-20230713_141945-za3j6zuh/files/checkpoint.pt
```

## Running LEON Models

For logging, we use [Weights and Biases](https://wandb.ai/site). You will be asked about your API code when first running an experiment with WandB - either sign up and create an account. You should also manually change the `entity` parameter of the `wandb.init` call in the `leon/train_job.py` file, which is currently set to `FILL_IN_YOUR_WANDB_ENTITY_HERE`.

**Note:** We only changed the ``train_job.py`` to run JOB and have not added WandB logging to the ``train_tpch.py`` file. Addtionally, we added ``train_stack.py`` for STACK.

To install the required libraries for LEON, follow the instructions:

```bash
docker exec -it balsa bash

cd /app/leon
. activate_env.sh
pip install -r requirements.txt
```

### Training a LEON Model

Similar as in Balsa, we also added the train/test splits as experiment parameters to the `train_job.py` file. You can add additional experiments in the `get_train_test_split` method of the `train_job.py` script or add the train/test splits in `leon/util/train_test_splits.py`.

```bash
docker exec -it balsa bash

cd /app/leon
. activate_env.sh

python3 train_job.py --wandb --experiment RandomSplit1
```

### Evaluating a LEON Model

Model checkpoints of LEON are saved in folders of the pattern `leon/<experiment>_<timestamp>`, for example `RandomSplit1_0807-141559`. LEON uses multiple models and as such has multiple checkpoint files for each model. We use the best train model(s) for evaluation.

```bash
docker exec -it balsa bash

cd /app/leon
. activate_env.sh

python3 test_job.py --log_file_path log__random_split_1.csv --experiment RandomSplit1 --model_path RandomSplit1_0807-141559 --logs_name RandomSplit1
```

For STACK:

```bash
python3 test_stack.py --log_file_path log__stack__random_split_1.csv --experiment STACK_RandomSplit1 --model_path STACK_RandomSplit1_1127-084510 --logs_name STACK_RandomSplit1
python3 test_stack.py --log_file_path log__stack__base_query_split_1.csv --experiment STACK_BaseQuerySplit1 --model_path STACK_BaseQuerySplit1_1121-211734 --logs_name STACK_BaseQuerySplit1
python3 test_stack.py --log_file_path log__stack__leave_one_out_split_1.csv --experiment STACK_LeaveOneOutSplit1 --model_path STACK_LeaveOneOutSplit1_1203-092739 --logs_name STACK_LeaveOneOutSplit1
```


## Running HybridQO Models

The HybridQO Model assumes Python version `3.10`, since the `psqlparse` library cannot be installed with `3.11`. It is fine if the base python of Ubuntu uses a different version, as long as your create the `venv` environment with Python `3.10`.

For logging, we use [Weights and Biases](https://wandb.ai/site). You will be asked about your API code when first running an experiment with WandB - either sign up and create an account. You should also manually change the `entity` parameter of the `wandb.init` call in the `hybrid_qo/run_mcts.py` file, which is currently set to `FILL_IN_YOUR_WANDB_ENTITY_HERE`.

### Training a HybridQO Model

> **WARNING**: Make sure, that you change the `hybrid_qo/ImportantConfig.py` file, such that you are running against your desired workload.

```bash
docker exec -it balsa bash

cd /app/hybrid_qo
# Make sure it is using Python 3.10!
python3.10 -m venv env
source env/bin/activate

pip install -r requirements.txt


# Make sure the hybrid_qo/ImportantConfig.py file contains the config that you want to run
# (i.e. queries_file, n_epochs and database entries especially)

python run_mcts.py
```

### Evaluating a HybridQO Model

HybridQO already runs the test set every 5 epochs. As such, you can extract the logs and use the values at the specified epoch after your training run concluded. Logs are automatically saved under `hybrid_qo/logs`.

