> This repository has been taken from [HybridQO / HyperQO](https://github.com/yxfish13/HyperQO/tree/main). Please refer to the original repository and cite their [publication](https://dl.acm.org/doi/abs/10.14778/3565838.3565846).

# HyperQO


# Development
    This is the code for the hyperQO: "Cost-based or Learning-based? A Hybrid Query Optimizer for Query Plan Selection" paper. This is an experimental version, and we will release a code in the form of a PostgreSQL plug-in in the future to realize parallel computing and reduce planning overhead.

# Requirements
- Pytorch 1.0
- Python 3.7
- torchfold
- psqlparse

## Install the PostgreSQL and pg_hint_plan
We made some fixes to pg_hint_plan to better support the leading hint of prefixes. The PostgreSQL and pg_hint_plan is here[https://github.com/yxfish13/PostgreSQL12.1_hint].
### 1. Install PostgreSQL
    ```sh
    cd postgresql-12.1/
    ./configure --prefix=/usr/local/pgsql --with-segsize=16 --with-blocksize=32 --with-wal-segsize=64 --with-wal-blocksize=64 --with-libedit-preferred  --with-python --with-openssl --with-libxml --with-libxslt --enable-thread-safety --enable-nls=en_US.UTF-8
    make
    make install
    ```
### 2. Install pg_hint_plan
    ```sh
    cd postgresql-12.1/pg_hint_plan-REL12_1_3_6/
    make
    make install
    ```

## Running
1. configurate the ImportantConfig.py
2. run
    ```sh
        python3 run_mcts.py
    ```

