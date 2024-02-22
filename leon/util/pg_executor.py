import collections
import contextlib
import socket

import psycopg2
import psycopg2.extensions
# import ray
from psycopg2.extensions import POLL_OK, POLL_READ, POLL_WRITE
from select import select
import os
import time

# Change these strings so that psycopg2.connect(dsn=dsn_val) works correctly
# for local & remote Postgres.

# # JOB/IMDB.
# LOCAL_DSN = "postgres://postgres:postgres@pg_balsa/imdbload"
# REMOTE_DSN = "postgres://postgres:postgres@pg_balsa/imdbload"

# STACK
LOCAL_DSN = "postgres://postgres:postgres@pg_balsa/so"
REMOTE_DSN = "postgres://postgres:postgres@pg_balsa/so"

# database = "imdbload"
database = "so"
user = "postgres"
password = "postgres"
host = "pg_balsa"
port = "5432"

QUERY_LOG_FILE = 'query_log_file.txt'
NUM_EXECUTIONS = 3




# TPC-H.
# LOCAL_DSN = "postgres://psycopg:psycopg@localhost/tpch-sf10"
# REMOTE_DSN = "postgres://psycopg:psycopg@localhost/tpch-sf10"

# A simple class holding an execution result.
#   result: a list, outputs from cursor.fetchall().  E.g., the textual outputs
#     from EXPLAIN ANALYZE.
#   has_timeout: bool, set to True iff an execution has run outside of its
#     allocated timeouts; False otherwise (e.g., for non-execution statements
#     such as EXPLAIN).
#   server_ip: str, the private IP address of the Postgres server that
#     generated this Result.
Result = collections.namedtuple(
    'Result',
    ['result', 'has_timeout', 'server_ip'],
)


# ----------------------------------------
#     Psycopg setup
# ----------------------------------------


def wait_select_inter(conn):
    while 1:
        try:
            state = conn.poll()
            if state == POLL_OK:
                break
            elif state == POLL_READ:
                select([conn.fileno()], [], [])
            elif state == POLL_WRITE:
                select([], [conn.fileno()], [])
            else:
                raise conn.OperationalError("bad state from poll: %s" % state)
        except KeyboardInterrupt:
            conn.cancel()
            # the loop will be broken by a server error
            continue


# psycopg2.extensions.set_wait_callback(wait_select_inter)


@contextlib.contextmanager
def Cursor(dsn=LOCAL_DSN):
    """Get a cursor to local Postgres database."""
    # TODO: create the cursor once per worker node.
    def get_connection(dsn):
        try:
            return psycopg2.connect(dsn)
        except psycopg2.OperationalError:
            time.sleep(10)
            return get_connection(dsn)

    conn = get_connection(dsn)
    conn.set_session(autocommit=True)
    try:
        with conn.cursor() as cursor:
            cursor.execute("load 'pg_hint_plan';")
            yield cursor
    finally:
        conn.close()


# ----------------------------------------
#     Postgres execution
# ----------------------------------------


def _SetGeneticOptimizer(flag, cursor):
    # NOTE: DISCARD would erase settings specified via SET commands.  Make sure
    # no DISCARD ALL is called unexpectedly.
    assert cursor is not None
    assert flag in ['on', 'off', 'default'], flag
    cursor.execute('set geqo = {};'.format(flag))
    assert cursor.statusmessage == 'SET'


def ExecuteRemote(sql, verbose=False, geqo_off=False, timeout_ms=None):
    return _ExecuteRemoteImpl.remote(sql, verbose, geqo_off, timeout_ms)


# @ray.remote(resources={'pg': 1})
def _ExecuteRemoteImpl(sql, verbose, geqo_off, timeout_ms):
    with Cursor(dsn=REMOTE_DSN) as cursor:
        return Execute(sql, verbose, geqo_off, timeout_ms, cursor)


def Execute(sql, verbose=False, geqo_off=False, timeout_ms=None, cursor=None, file_prefix=''):
    """Executes a sql statement.

    Returns:
      A pg_executor.Result.
    """
    if verbose:
        print(sql)

    _SetGeneticOptimizer('off' if geqo_off else 'on', cursor)
    if timeout_ms is not None:
        cursor.execute('SET statement_timeout to {}'.format(int(timeout_ms)))
    else:
        # Passing None / setting to 0 means disabling timeout.
        cursor.execute('SET statement_timeout to 0')
    try:
        for iteration in range(NUM_EXECUTIONS):
            cursor.execute(sql)
            result = cursor.fetchall()
        has_timeout = False
    except Exception as e:
        if isinstance(e, psycopg2.errors.QueryCanceled):
            assert 'canceling statement due to statement timeout' \
                   in str(e).strip(), e
            result = []
            has_timeout = True
        elif isinstance(e, psycopg2.errors.InternalError_):
            print(
                'psycopg2.errors.InternalError_, treating as a' \
                ' timeout'
            )
            print(e)
            result = []
            has_timeout = True
        elif isinstance(e, psycopg2.OperationalError):
            # This usually indicates an expensive query, putting the server
            # into recovery mode.  'cursor' may get closed too.
            print('Treating as a timeout:', e)
            result = []
            has_timeout = True

            if 'the database system is in recovery mode' in str(e).strip():
                time.sleep(10)

            # COMMENTED OUT, since it leads unrecoverable errors in case the database
            # goes down shortly because of an erroneous query
            #
            # if 'SSL SYSCALL error: EOF detected' in str(e).strip():
            #     # This usually indicates an expensive query, putting the server
            #     # into recovery mode.  'cursor' may get closed too.
            #     print('Treating as a timeout:', e)
            #     result = []
            #     has_timeout = True
            # else:
            #     # E.g., psycopg2.OperationalError: FATAL: the database system
            #     # is in recovery mode
            #     raise e
        else:
            raise e
    else:
        # Add logging of executed queries
        #
        try:
            latency = float(result[0][0][0]['Execution Time'])
            planning_time = float(result[0][0][0]['Planning Time'])

        except IndexError:
            pass
        except KeyError:
            pass
        else:
            query_log_file_name = "_".join([file_prefix, QUERY_LOG_FILE])

            with open(query_log_file_name, 'a') as qlf:
                qlf.write(";".join([str(sql).replace('\n',' '), str(latency), str(planning_time), str(timeout_ms), str(geqo_off)]))
                qlf.write(os.linesep)

            print(f"Executed Query on Postgres taking {latency:.3f} ({planning_time:.3f} planning).")
    
    try:
        _SetGeneticOptimizer('default', cursor)
    except psycopg2.InterfaceError as e:
        # This could happen if the server is in recovery, due to some expensive
        # queries just crashing the server (see the above exceptions).
        assert 'cursor already closed' in str(e), e
        pass
    ip = socket.gethostbyname(socket.gethostname())
    return Result(result, has_timeout, ip)
