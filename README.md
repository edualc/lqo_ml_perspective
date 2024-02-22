# Is Your Learned Query Optimizer Behaving As You Expect? - A Machine Learning Perspective

This repository includes the code base used in the paper "Is Your Learned Query Optimizer Behaving As You Expect? A Machine Learning Perspective", accepted at VLDB2024, the 50th International Conference on Very Large Databases and available as a preprint on [arXiv](https://arxiv.org/abs/2309.01551).


## Authors

* Claude Lehmann; Zurich University of Applied Sciences; Winterthur, Switzerland; [claude.lehmann@zhaw.ch](claude.lehmann@zhaw.ch)
* Pavel Sulimov; Zurich University of Applied Sciences; Winterthur, Switzerland; [pavel.sulimov@zhaw.ch](pavel.sulimov@zhaw.ch)
* Kurt Stockinger; Zurich University of Applied Sciences; Winterthur, Switzerland; [kurt.stockinger@zhaw.ch](kurt.stockinger@zhaw.ch)


# Setup

This repository holds a variety of learned query optimizer methods and ways to train and evaluate them. All methods are run inside docker container environments using **64 GB of RAM**, **16 CPU cores** and with **access to a GPU** (in our case, NVIDIA Tesla T4). Please refer to the individual `README.md` files in the `docker` subdirectory on how to setup each individual method.

We assume that the JOB and STACK datasets has been downloaded and prepared in the `data` subfolder. See the [data/README.md](data/README.md) on how to do this.


## Train/Test Splits

The train/test splits can be found in the following places:
- [Balsa/Neo](balsa/experiments.py) (see the test set definitions at the beginning of the file, for example `RANDOM_SPLIT_1__TEST_QUERIES`)
- [Bao](bao/queries) (as explicit directories containing just the SQL files)
- [LEON](leon/util/train_test_splits.py) (Note, that LEON references the queries from the queries directory of Balsa `balsa/queries`, see the definition of the `load_sql_Files` method in `leon/train_job.py` and `leon/train_stack.py`)
- [HybridQO](hybrid_qo/workload) (Splits are defined as specific JSON files referencing the queries directly, IPython notebooks are given in `notebooks/Build_HybridQO_Workloads.ipynb` and `notebooks/Build_HybridQO_Workloads__STACK.ipynb` on how to prepare these workload files.)

## Methods

Information about setting up and running all the methods included in this codebase (namely Neo, Bao, Balsa, LEON and HybridQO) can be found in the corresponding READMEs of the docker directories.

Currently we include the following methods:
- Neo: see the [docker/postgres_balsa/README.md](docker/postgres_balsa/README.md) (Neo is run through the code base of Balsa, as the Neo authors did not make the code available. However, the authors of Neo have assured us, that the implementation of Neo inside the Balsa codebase has been checked and approved by them.)
- Bao: see the [docker/postgres_bao/README.md](docker/postgres_bao/README.md) (Bao uses an extension running inside PostgreSQL. To make sure it does not interfere with other methods, Bao is run in a seperate container with identical settings otherwise)
- Balsa: see the [docker/postgres_balsa/README.md](docker/postgres_balsa/README.md)
- LEON: see the [docker/postgres_balsa/README.md](docker/postgres_balsa/README.md) to set up the container (LEON runs inside the same containers as Balsa)
- HybridQO: see the [docker/postgres_balsa/README.md](docker/postgres_balsa/README.md) to set up the container (HybridQO runs inside the same containers as Balsa)

> **Note:** The PostgreSQL baseline is run from the Bao codebase using the `bao/run_test_queries.py` with the parameter `--use_postgres`.

## Figures

The figures used in our publication are created using the notebooks found in the `notebooks` directory and the data from the `experiment_logs` directory (though the individual model checkpoints and log files are generally available in the respective subdirectories of the methods). The notebooks `notebooks/FullEvaluation_<workload>_MergeResultFiles.ipynb` are meant to collect the individual log files for all experiments, while the `notebooks/FullEvaluation_<workload>_ResultAnalysis.ipynb` contains code for the detailed result figures.

If you want to run these notebooks, you have to install the dependencies on the outermost `requirements.txt` file in order to have all libraries available:

```bash
python3 -m venv env

. activate_env.sh
pip install -r requirements.txt

jupyter-notebook
```


# Citations

Since we include the code bases from recent publications, please make sure to also include their citations. We thank the authors of the previous work for making their research available:

>Marcus, Ryan, et al. "**Neo: A Learned Query Optimizer.**" Proceedings of the VLDB Endowment 12.11.

>Marcus, Ryan, et al. "**Bao: Making learned query optimization practical.**" Proceedings of the 2021 International Conference on Management of Data. 2021.

>Yang, Zongheng, et al. "**Balsa: Learning a Query Optimizer Without Expert Demonstrations.**" Proceedings of the 2022 International Conference on Management of Data. 2022.

>Chen, Xu, et al. "**LEON: a new framework for ml-aided query optimization.**" Proceedings of the VLDB Endowment 16.9 (2023): 2261-2273.

>Yu, Xiang, et al. "**Cost-based or learning-based? A hybrid query optimizer for query plan selection.**" Proceedings of the VLDB Endowment 15.13 (2022): 3924-3936.

Additionally, we use the Join Order Benchmark published by Leis et al.:

>Leis, Viktor, et al. "**How good are query optimizers, really?.**" Proceedings of the VLDB Endowment 9.3 (2015): 204-215.

And the STACK benchmark published by Marcus et al.:

>Marcus, Ryan, et al. "**Bao: Making learned query optimization practical.**" Proceedings of the 2021 International Conference on Management of Data. 2021.
