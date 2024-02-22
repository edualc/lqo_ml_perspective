> This repository has been taken from [Bao](https://github.com/learnedsystems/BaoForPostgreSQL). Please refer to the original repository and cite their [publication](https://dl.acm.org/doi/abs/10.1145/3448016.3452838).


# Bao

## README

### Installation

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

After installation, you can always enter the environment using the `activate_env.sh` scirpt.

### Changelog from original repository

- Added more logging
- Dockerized the setup
- Changed the postgres config to use the same parallelized setup that the Balsa paper used
- Changed the timeout of queries sent to Postgres
    - Timeout increased from 5min to 15min
    - In case of a timeout or an error, we set the execution time to twice the timeout limit



### Original README from the [author's repository](https://github.com/learnedsystems/BaoForPostgreSQL)


```
Marcus, R., Negi, P., Mao, H., Tatbul, N., Alizadeh, M. and Kraska, T., 2022. Bao: Making learned query optimization practical. ACM SIGMOD Record, 51(1), pp.6-13.
```


This is a prototype implementation of Bao for PostgreSQL. Bao is a learned query optimizer that learns to "steer" the PostgreSQL optimizer by issuing coarse-grained query hints. For more information about Bao, [check out the paper](https://rm.cab/bao).

Documentation, including a tutorial, is available here: https://rmarcus.info/bao_docs/

While this repository contains working prototype implementations of many of the pieces required to build a production-ready learned query optimizer, this code itself should not be used in production in its current form. Notable limitations include:

* The reward function is currently restricted to being a user-supplied value or the query latency in wall time. Thus, results may be inconsistent with high degrees of parallelism.
* The Bao server component does not perform any level of authentication or encryption. Do not run it on a machine directly accessible from an untrusted network.
* The code has not been audited for security issues. Since the PostgreSQL integration is written using the C hooks system, there are almost certainly issues.

This software is available under the AGPLv3 license. 
