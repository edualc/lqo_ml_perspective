## Installing Postgres + Bao

In this directory you find all the relevant files for running Postgres + [Bao](https://github.com/learnedsystems/BaoForPostgreSQL) in a docker environment.

>Marcus, Ryan, et al. "**Bao: Making learned query optimization practical.**" Proceedings of the 2021 International Conference on Management of Data. 2021.


### Installation

The JOB database files are to be placed into the `<repo>/data/` directory, outside the `docker/` directory (see [Readme](../../data/README.md)). It contains the basic steps for downloading and preparing the JOB data for insertion into Postgres. The steps below assume, you already downloaded the JOB database and performed the preprocessing steps.


#### Building Postgres + Bao containers

To distinguish the different containers, we call the Postgres container `pg_bao` and the Bao server component `bao_server`. The docker images have an added -`_img` suffix, i.e. `pg_bao_img` and `bao_server_img`.


The following script `generate_docker_containers.sh` is available to build the containers:
```bash
    cd /<repo>/docker/postgres_bao
    . generate_docker_containers.sh
```


#### Initializing the Postgres container

In our case, we assume that the `imdbload` database is created by setting the `POSTGRES_DB` environment parameter in the [docker-compose.yml](docker-compose.yml). The process of loading the JOB data into Postgres is taken from the [Balsa](https://github.com/balsa-project/balsa) repository.

```bash
# Start the container(s)
docker-compose up

# Open a different shell, since the one running docker-compose shows you the log output. If you want to run the container detached, add -d or --detach after docker-compose up, i.e. docker-compose up -d
#
# Start interactive shell in PG container
docker exec -it pg_bao bash

# Change user to Postgres (linux user password is set to "postgres" in Dockerfile)
su postgres


# PG_BAO EXTENSION
# ==========================================
#
# Install the PG Extension for Bao (the sudo password for the linux postgres user is postgres)
cd /app/bao/pg_extension
sudo make USE_PGXS=1 install

# Add config option to load bao (already done in provided postgresql.conf file)
# --> just copy the postgresql.conf file
# echo "shared_preload_libraries = 'pg_bao'" >> /var/lib/postgresql/data/postgresql.conf
#
# Make sure, that enough memory is available, otherwise the container might not start again
# 
cp /app/postgresql.conf /pgdata/postgresql.conf

# Restart Postgres instance (the container will stop)
pg_ctl restart

# Restart the container by stopping the docker-compose command in your other shell
docker-compose up

# Reenter pg_bao in your second shell
docker exec -it pg_bao bash
su postgres

# LOAD JOB INTO DB
# ==========================================
#
# Run script to load the CSVs into the database. This process takes on the order of 15 minutes.
bash data/imdb/load-postgres/load_job_postgres.sh /data/imdb

# LOAD STACK INTO DB (Takes ~2.5h to restore dump, ~16min to ANALYZE)
# ==========================================
#
psql -c "CREATE DATABASE so;"
pg_restore -d so -j 4 --clean --no-privileges --no-owner --verbose data/stack/so_pg12
psql -d so -c "ANALYZE verbose;"
```


##### Checking for a successful installation of the Bao extension

```bash
# Start the containers, if not already running
cd /<repo>/docker/postgres_bao
docker-compose up

# Enter the pg_bao container
docker exec -it pg_bao bash

# Switch to the postgres user
su postgres

# Enter Postgres and check if the "enable_bao" parameter is available
psql
show enable_bao;
```

The output should look something like this:

```bash
postgres@0225566cecbb:/$ psql
psql (12.5 (Debian 12.5-1.pgdg100+1))
Type "help" for help.

postgres=# show enable_bao;
enable_bao
------------
off
(1 row)
```


##### Setting the correct configuration

We opted for static IPs to be set inside the `docker-compose-yml`, there the postgres instance is available under `10.5.0.5` and the bao_server under `10.5.0.6`. The host names generated through container names worked for everything but the extension of Bao inside Postgres. However, setting this option in the Postgres config makes the container fail to boot again, due to the order of loading. Instead, we specifically set the `bao_host` option during every session, similar to the `enable_bao` parameter.


#### Initializing Bao Server Container

```bash
docker-compose up

docker exec -it bao_server bash

cd /app/bao

# Generate a python3 environment. In case there is an error and the env folder already exists,
# try removing the env folder before running the command again
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

cd bao_server
python3 main.py
```


### Starting the containers and training a model

In order to set up and start everything, we assume the containers are running on the same machine but in seperate Docker containers (`pg_bao` and `bao_server`). Given that some scripts have to be kept running, we will be using `screen` and `tmux`, to keep the shells open and visible for checking logs/outputs. You might have to install tmux (`apt-get install tmux`).

```bash
# Open the first screen
screen -S bao_compose
docker-compose up
```

In case the `pg_bao` container is not starting because the `pg_bao` file cannot be read/is not found, simply change the `docker/postgres_bao/pgdata/postgresql.conf` from outside the container and comment out the last line (preloading the `pg_bao` extension). Enter the container with `docker exec -it pg_bao bash` and install the `pg_extension` again, as done above. Uncomment the line in the `postgresql.conf` file again and restart the container - this should let it start again.

```bash
# Open another screen
screen -S bao_server
docker exec -it bao_server bash

# Open a tmux session inside the screen
tmux new -s bao_main
cd /app/bao
. activate_env.sh
cd bao_server/
python3 main.py

# Use Ctrl+b d to exit the tmux session (you can use tmux a -t <session_name> to reattach)
Ctrl+b d

# Open a new tmux session inside the screen again
tmux new -s bao_train

# Start a training run
cd /app/bao
. activate_env.sh
# Training Bao

python3 run_queries.py --query_dir queries/job__base_query_split_1/train --output_file logs/train__bao__base_query_split_1.txt

# Testing Bao
python3 run_test_queries.py --use_bao --query_dir queries/join-order-benchmark --output_file logs/test__bao__job.txt

# Testing Postgres
python3 run_test_queries.py --use_postgres --query_dir queries/join-order-benchmark --output_file logs/test__postgres__job.txt

# Use Ctrl+b d to exit the tmux session
Ctrl+b d
```
