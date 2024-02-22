### Data Directory

This directory `data/` holds the data of the various databases. It is used to be mounted through Docker volumes.

#### IMDB

In accordance to how [Balsa](https://github.com/balsa-project/balsa) loads the IMDB data, use the `data/imdb/` folder to put the CSVs in. The `data/preprend_imdb_headers.py` file and the `data/requirements.txt` are set up, so that the script can be used.

```bash
# Switch into the data directory
cd /<repo>/data

# Create a small python env to use the scripts below
python3 -m venv env
. activate_env.sh
pip install -r requirements.txt

# Download JOB files into imdb directory
cd imdb
wget -c http://homepages.cwi.nl/~boncz/job/imdb.tgz && tar -xvzf imdb.tgz

# Prepend headers to CSV files
cd ..
python prepend_imdb_headers.py

# exit env
deactivate
```


#### STACK

This dataset was published in the [Bao paper](https://dl.acm.org/doi/pdf/10.1145/3448016.3452838) and is available [here](https://rmarcus.info/stack.html). We use the PostgreSQL 12 dump. See the [Balsa](../docker/postgres_balsa/README.md) & [Bao](../docker/postgres_balsa/README.md) READMEs on how to install them specifically.

```bash
cd /<repo>/data/stack

# Download PG Dump from Ryan Marcus
wget https://www.dropbox.com/s/98u5ec6yb365913/so_pg12
```

##### STACK Queries

(This step is **not required**, as we already provide the selected subset of STACK queries - but if you want to look at all queries for completeness:)

```bash
cd /<repo>/data/stack

wget https://rmarcus.info/so_queries.tar.zst

sudo apt-get install zstd tar
unzstd so_queries.tar.zst
tar -xvf so_queries.tar
```
