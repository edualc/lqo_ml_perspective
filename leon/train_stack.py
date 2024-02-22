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

from tqdm.auto import tqdm
import argparse
import wandb

def getexpnum(exp):
    num = 0
    for i in exp:
        num = num + len(i)
    return num


def seed_torch(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def getTrainPair(output1, output2, trainpair):
    for i in range(0, len(output1)):
        if len(output1[i]) == 0 or len(output2[i]) == 0:
            continue
        for j in output1[i]:
            for k in output2[i]:
                if (j[2] == k[2]) and (j[1] == k[1]):
                    #  print('equal')
                    continue
                if (j[3] == k[3]):
                    #  print('equal')
                    continue
                if (j[6] != k[6]):
                    continue
                tem = []
                # encoding
                tem.append(j[4])
                # latency
                tem.append(j[3])
                # cost
                tem.append(j[0])
                tem.append(k[4])
                tem.append(k[3])
                tem.append(k[0])
                trainpair[i].append(tem)


def getBestTrainPair(output1, output2):
    trainpair = [[] for _ in range(20)]
    for i in range(0, len(output1)):
        if len(output1[i]) == 0 or len(output2[i]) == 0:
            continue
        for j in output1[i]:
            for k in output2[i]:
                if (j[2] == k[2]) and (j[1] == k[1]):
                    #  print('equal')
                    continue
                if (j[3] == k[3]):
                    #  print('equal')
                    continue
                tem = []
                # encoding
                tem.append(j[4])
                # latency
                tem.append(j[3])
                # cost
                tem.append(j[0])
                tem.append(k[4])
                tem.append(k[3])
                tem.append(k[0])
                trainpair[i].append(tem)
    return trainpair


def getPartTrainPair(alltrainpair, rate=0.2):
    if len(alltrainpair) == 0:
        return []
    sortlist = sorted(alltrainpair, key=lambda x: x[6], reverse=True)
    alltrainpair = sortlist[int(len(alltrainpair) * rate):]
    return sortlist[0:int(len(alltrainpair) * rate)]


def calculateLossForBatch(latencies: list, costs: list, calibration: torch.Tensor):
    """
    :param latencies: real latency
    :param costs: PG estimated cost
    :param calibration: ML model's calibration for cost
    :return: loss to learn the ML model
    """
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    calibration = calibration.view(-1, 2)
    costs = torch.tensor(costs, device=DEVICE).view(-1, 2)
    calibratedCost = calibration * costs
    softm = nn.functional.softmax(calibratedCost, dim=1)
    assert (2 * len(costs) == len(latencies)) and (len(latencies) % 2 == 0)
    res = []
    for i in range(0, len(latencies), 2):
        if latencies[i] > latencies[i + 1]:
            res.append(0)
        else:
            res.append(1)
    res = torch.tensor(res, device=DEVICE)
    return loss_fn(softm, res)


def calculateLossForBatch_MR_0(latencies: list, costs: list, calibration: torch.Tensor):
    """
    :param latencies: real latency
    :param costs: PG estimated cost
    :param calibration: ML model's calibration for cost
    :return: loss to learn the ML model
    """
    loss_fn = nn.MarginRankingLoss()
    calibration = calibration.view(-1, 2)
    costs = torch.tensor(costs, device=DEVICE).view(-1, 2)
    calibratedCost = calibration * costs
    c1, c2 = torch.chunk(calibratedCost, 2, dim=1)
    c1 = torch.squeeze(c1)
    c2 = torch.squeeze(c2)
    assert (2 * len(costs) == len(latencies)) and (len(latencies) % 2 == 0)
    res = []
    for i in range(0, len(latencies), 2):
        if latencies[i] > latencies[i + 1]:
            res.append(1)
        else:
            res.append(-1)
    res = torch.tensor(res, device=DEVICE)
    return loss_fn(c1, c2, res)


def geometric_mean(data):  # 计算几何平均数
    total = 1
    for i in data:
        total *= i  # 等同于total=total*i
    return pow(total, 1 / len(data))


def get_logger(filename, verbosity=1, name=None):
    filename = filename + '.txt'
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def load_sql_Files(sql_list: list):
    """
    :param sql_list: list of sql template name
    :return: list of path of sql query file path
    """
    sqllist = []
    for i in range(0, len(sql_list)):
        sqlFiles = './../balsa/queries/stack/' + sql_list[i] + '.sql'
        if not os.path.exists(sqlFiles):
            raise IOError("File Not Exists!")
        sqllist.append(sqlFiles)
    return sqllist


def load_sql(sql_list: list):
    """
    :param sql_list: list of sql file path
    :return: list of sql query string
    """
    sqls = []
    for i in sql_list:
        with open(i, 'r') as f:
            data = f.read().splitlines()
            # Remove lines that are just comments starting with "--"
            data = [line for line in data if not line.startswith('--')]

            sql = ' '.join(data)
        sqls.append(sql)
        f.close()
    return sqls


def collects(finnode, workload, exp, timeout):
    allPlans = [finnode]
    # print('collect')
    currentChild = finnode
    temlevel = currentChild.info.get("currentLevel")
    if (not temlevel == None) and temlevel > 1:

        temsql = currentChild.to_sql(currentChild.info["join_conds"],
                                     with_select_exprs=True)

        temhint = currentChild.hint_str()
        found = False
        for i in exp[temlevel]:
            if (i[1] == temsql and i[2] == temhint):
                found = True
                break
        if not found:
            tem = []
            tem.append(math.log(currentChild.info["cost"]))
            tem.append(temsql)
            tem.append(temhint)
            nodelatency = currentChild.info.get("latency")
            if nodelatency == None:
                nodelatency = postgres.GetLatencyFromPg(temsql, temhint, verbose=False, check_hint_used=False,
                                                        timeout=timeout, dropbuffer=False)
                tem.append(nodelatency)
                tem.append([currentChild.info["encoding"], currentChild.info["node"]])
                tem.append(currentChild)
                tem.append(currentChild.info["join_ids"])
                exp[temlevel].append(copy.deepcopy(tem))
            else:
                tem.append(nodelatency)
                tem.append([currentChild.info["encoding"], currentChild.info["node"]])
                tem.appedn(currentChild)
                tem.append(currentChild.info["join_ids"])
                exp[temlevel].append(copy.deepcopy(tem))
    while (allPlans):
        currentNode = allPlans.pop()
        allPlans.extend(currentNode.children)
        for currentChild in currentNode.children:
            temlevel = currentChild.info.get("currentLevel")
            # print(temlevel)
            if (not temlevel == None) and temlevel > 1:
                #  print(currentChild)
                temsql = currentChild.to_sql(currentChild.info["join_conds"],
                                             with_select_exprs=True)
                #    print(temsql)
                temhint = currentChild.hint_str()
                found = False
                for i in exp[temlevel]:
                    if (i[1] == temsql and i[2] == temhint):
                        found = True
                        break
                if not found:
                    tem = []
                    tem.append(math.log(currentChild.info["cost"]))
                    tem.append(temsql)
                    tem.append(temhint)
                    nodelatency = currentChild.info.get("latency")
                    if nodelatency == None:
                        nodelatency = postgres.GetLatencyFromPg(temsql, temhint, verbose=False, check_hint_used=False,
                                                                timeout=timeout, dropbuffer=False)
                        tem.append(nodelatency)
                        tem.append([currentChild.info["encoding"], currentChild.info["node"]])
                        tem.append(currentChild)
                        tem.append(currentChild.info["join_ids"])
                        exp[temlevel].append(copy.deepcopy(tem))
                    else:
                        tem.append(nodelatency)
                        tem.append([currentChild.info["encoding"], currentChild.info["node"]])
                        tem.append(currentChild)
                        tem.append(currentChild.info["join_ids"])
                        exp[temlevel].append(copy.deepcopy(tem))


def getGMRL(sqls, modellist, pg_latency, nodeFeaturizer, costCache, workload, exp=None, old=None):
    sql_ = load_sql(load_sql_Files(sqls))
    hints = []
    alllatency = []
    nodes = []
    for i in sqls:
        join_graph, all_join_conds, query_leaves, origin_dp_tables = DP.getPreCondition(
            './../balsa/queries/stack/' + i + '.sql')
        # TEST_left_prune_bayes
        bestplanhint, finnode = DP.dp.TEST_left_prune_bayes(join_graph, all_join_conds, query_leaves, origin_dp_tables,
                                                            workload,
                                                            modellist, nodeFeaturizer, costCache)
        hints.append(bestplanhint)
        nodes.append(finnode)
    for i in range(0, len(sql_)):
        latency = postgres.GetLatencyFromPg(sql_[i], hints[i], verbose=False, check_hint_used=False,
                                            timeout=90000, dropbuffer=dropbuffer)
        print(sqls[i], latency, pg_latency[i], latency / pg_latency[i])
        alllatency.append(latency / pg_latency[i])
    if old != None:
        for i in range(len(sqls)):
            if alllatency[i] > 1.4:
                print('degradation collect')
                collects(finnode, workload, exp, old[i])
    return geometric_mean(alllatency)


def getModels(maxLevel):
    modellist = ['blank', 'blank']
    optlist = ['blank', 'blank']
    for i in range(maxLevel - 1):
        modellist.append(treeconv_dropout.TreeConvolution(406, 87, 1).to(DEVICE))
        optlist.append(torch.optim.AdamW(modellist[i + 2].parameters(), lr=0.001))
    return modellist, optlist


def getModelsFromFile(maxLevel, modelpath, device=None):
    if device is None:
        device = DEVICE

    modellist = ['blank', 'blank']
    optlist = ['blank', 'blank']
    for i in range(2, maxLevel + 1):
        full_model_path = modelpath + str(i) + '.pth'

        if not os.path.exists(full_model_path):
            model = treeconv_dropout.TreeConvolution(406, 87, 1).to(device)
            optlist.append(torch.optim.AdamW(model.parameters(), lr=0.001))
        else:
            model = torch.load(full_model_path).to(device if torch.cuda.is_available() else 'cpu')
            optlist.append(torch.optim.AdamW(model.parameters(), lr=0.001))
            print(f'[DP-{i}]\tSuccessfully loaded checkpoint from {full_model_path}')
        modellist.append(model)
    return modellist, optlist


def setInitialTimeout(sqls: list, dropbuffer, testtime=3):
    """
    :param sqls: list of sql string
    :return: timeout list
    """
    timeoutlist = []
    for i in sqls:
        timeout = postgres.GetLatencyFromPg(i, None, verbose=False, check_hint_used=False, timeout=90000,
                                                    dropbuffer=dropbuffer)
        timeoutlist.append(round(timeout, 3))
    return timeoutlist


def getPG_latency(sqls):
    pg_latency = []
    for i in sqls:
        latency = postgres.GetLatencyFromPg(i, None, verbose=False, check_hint_used=False, timeout=90000,
                                                    dropbuffer=False)
        pg_latency.append(latency)
    return pg_latency


def getdistribution(data, model):
    # [dp_costs,dp_query_encodings,dp_nodes]
    dp_costs = data[0]
    dp_query_encodings = data[1]
    dp_nodes = data[2]
    query_feats = (torch.cat(dp_query_encodings, dim=0)).to(DEVICE)
    trees, indexes = TreeConvFeaturize(nodeFeaturizer, dp_nodes)
    if torch.cuda.is_available():
        trees = trees.to(DEVICE)
        indexes = indexes.to(DEVICE)
        torch_dpcosts = (torch.tensor(dp_costs)).to(DEVICE)
    costbais = []
    for i in range(10):
        costbais.append(torch.tanh(model(query_feats, trees, indexes).to(DEVICE)).add(1))
    costbais = torch.cat(costbais, 1)
    costbais_mean = torch.mean(costbais, dim=1)
    costlist = torch.mul(costbais_mean, torch_dpcosts)
    distri = costlist / (torch.tensor(0) - torch.sum(costlist))
    return distri


def getOriDistribution(leveldict, model, oridistribution):
    for k, v in leveldict.items():
        oridistribution[k] = getdistribution(v, model).detach()


def getKL(ori_dis, newdis_tir):
    # KL = scipy.stats.entropy(ori_dis, newdis_tir)
    KL = torch.nn.functional.kl_div(newdis_tir.softmax(dim=-1).log(), ori_dis.softmax(dim=-1), reduction='sum')
    return KL


def getKLreg(leveldict, model, oridistribution):
    reg = 0
    for k, v in leveldict.items():
        newdis_part = getdistribution(v, model)
        reg_part = getKL(oridistribution[k], newdis_part)
        reg = reg + reg_part
    return reg


def get_train_test_split(experiment_name):
    all_available_queries = [
        'q1__q1-009', 'q1__q1-031', 'q1__q1-035', 'q1__q1-067', 'q1__q1-075', 'q1__q1-098', 'q1__q1-099', 'q1__q1-100',
        'q2__q2-001', 'q2__q2-012', 'q2__q2-032', 'q2__q2-035', 'q2__q2-050', 'q2__q2-081', 'q2__q2-094', 'q2__q2-098',
        'q3__q3-018', 'q3__q3-040', 'q3__q3-043', 'q3__q3-046', 'q3__q3-066', 'q3__q3-068', 'q3__q3-086', 'q3__q3-099',
        'q4__q4-002', 'q4__q4-026', 'q4__q4-041', 'q4__q4-042', 'q4__q4-064', 'q4__q4-074', 'q4__q4-086', 'q4__q4-089',
        'q5__q5-015', 'q5__q5-032', 'q5__q5-041', 'q5__q5-052', 'q5__q5-059', 'q5__q5-077', 'q5__q5-079', 'q5__q5-082',
        'q6__q6-002', 'q6__q6-009', 'q6__q6-060', 'q6__q6-064', 'q6__q6-065', 'q6__q6-067', 'q6__q6-069', 'q6__q6-085',
        'q7__q7-034', 'q7__q7-036', 'q7__q7-047', 'q7__q7-077', 'q7__q7-082', 'q7__q7-085', 'q7__q7-095', 'q7__q7-099',
        'q8__q8-006', 'q8__q8-025', 'q8__q8-046', 'q8__q8-062', 'q8__q8-065', 'q8__q8-074', 'q8__q8-076', 'q8__q8-096',
        'q11__0ea8bacde0e13a4314466435cf49c8e685b39fb1', 'q11__33e1caf220e5bea2e592c82eede1c0427e2c2570',
        'q11__6c5cba419c5b7b02d431aeb5e766d775d812967a', 'q11__87c4bd0930b02a3361ac2e86c453db1fec60dc6b',
        'q11__9389f58853715321e2a60ad743f99fc365f040cb', 'q11__aa96c8d7abbf8a5b6d29473c1b9447a84f8b4f52',
        'q11__c1ae2a992cde4ea2c4922d852df22043254b4f84', 'q11__e4ca35591923cf4efc89e64b17fb4d330c0b34df',
        'q12__06c8d6886a03d4d92837f38ff395b888de007d33', 'q12__0700720596313f7fa30c0dd3d4a3001c896ba760',
        'q12__547c6bf1994c9b2ba82a7ae32f4b051beabf46fd', 'q12__55de941e8497cfeeb93d3f8f2d7a18489e0e6c32',
        'q12__5a5ff9bd9de9e748708116727803117e453e30da', 'q12__76a47868a09eec9f95bacb2cf21492d353698eb7',
        'q12__812a3effb91cb789490fc2e12af772b1a35f8552', 'q12__bde6c0cf5e67ddae4ec0dbb787291da703e406d5',
        'q13__13ad1b8c6bea4fda1892b9fa82cc1ceb9ceb85fc', 'q13__1ddcc8650e17b292bc7344902baffc90c5ae5761',
        'q13__935e2051bf80eeafe91aeb6eb719b6b64b9592c2', 'q13__a091adce62743b65c04532e98e8ff3d7e546ea77',
        'q13__a3d03772d880754fc4e150d82908757477ae2186', 'q13__add0df9dccb2790c14508e19c9e0deb79fad6ea2',
        'q13__d383cd5b4aee7d3f73508e2a1fe5f6d0f7dd42a2', 'q13__d4707be2adfdbc842f42acb1fc16e3a43faf7474',
        'q14__4063b6cbbd1c0f2a902a647aafe24174a75f53cd', 'q14__5dbc1d1f1a0467ad0086e6cb337144387a37533a',
        'q14__5e4835cd72aaa2d7be15b2a5ffa2e66156b3656f', 'q14__63c0776f1727638316b966fe748df7cc585a335b',
        'q14__719e692d411868ae7a93909757872d264f6bbf73', 'q14__74fd1af68d23f0690e3d0fc80bd9b42fa90a7e94',
        'q14__97e68ad5c2ced4c182366b3118a1f5f69b423fa6', 'q14__b49361f85785200ed6ec1f2eec357b7598c9e564',
        'q15__21e4988a3f47be288de5891d69acf91928ed94eb', 'q15__3e37e62655ceaebc14e79edad518e5710752f51d',
        'q15__543ab3f730e494a69e3d15e59675f491544cb15d', 'q15__78995a5fc0536aa53b99be32ce84dcbf40e826f3',
        'q15__b2ee2c788d30655058aeb992811e9a54f17f2998', 'q15__b8ddf65b0c0c7867a9b560e571d457fec410715c',
        'q15__c9619ad44302bada330d337c174f9dab77538622', 'q15__d5546c01928a687eb1f54e9f8eb4e1aff68fc381',
        'q16__1e863562a79ca1f7754c759ebab6a2addda0bde8', 'q16__374e3e4c9eefc294fa4c46220953336298df3622',
        'q16__b1a96cd48ba297dd93bce73c27b491069ad7449f', 'q16__d5290889129fb8e625f2b36fa106e30d6c4b243b',
        'q16__ea9efde510227beb8d624b8c4a6941b9d5e6e637', 'q16__ed2ffeaefcf5ad8bbadc713ccc766541e12080aa',
        'q16__f67cec3d635586efb847c832072be83b42cc45b7', 'q16__fbe34e8fdf672a34fd82cbbd6d9a81fd02ce17d1',
    ]

    if experiment_name == 'STACK_LeaveOneOutSplit1':
        test_queries = STACK__LEAVE_ONE_OUT_SPLIT_1__TEST_QUERIES
    
    elif experiment_name == 'STACK_BaseQuerySplit1':
        test_queries = STACK__BASE_QUERY_SPLIT_1__TEST_QUERIES
    
    elif experiment_name == 'STACK_RandomSplit1':
        test_queries = STACK__RANDOM_SPLIT_1__TEST_QUERIES

    elif experiment_name == 'DebugInterval':
        train_queries = ['q7__q7-099']
        test_queries = ['q4__q4-026']
        return train_queries, test_queries

    else:
        raise NotImplementedError(f"Split {experiment_name} could not be found.")
    
    test_queries = [f.split('.sql')[0] for f in test_queries]

    train_queries = []
    for q in all_available_queries:
        if q not in test_queries:
            train_queries.append(q)

    return train_queries, test_queries


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, help='Defines which train-test split to use (see #get_train_test_split method).')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--no_wandb', dest='wandb', action='store_false')

    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = define_args()

    from util.pg_executor import LOCAL_DSN
    if not LOCAL_DSN.endswith('so'):
        raise ValueError(f"The current connection in 'util/pg_executor.py' does not point to the STACK database.")
    
    wandb.init(
        project='leon',
        entity='FILL_IN_YOUR_WANDB_ENTITY_HERE',
        save_code=False,
        config={ 'experiment': args.experiment, 'workload': 'STACK' }
    )

    logs_name = args.experiment
    ISOTIMEFORMAT = '%m%d-%H%M%S'
    config = {'log_path': ''}
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    loglogs = '_'.join((logs_name, timestamp))
    log_dir = os.path.join(config['log_path'], loglogs)
    os.makedirs(log_dir)
    log_file_name = os.path.join(log_dir, "running_log")
    logger = get_logger(log_file_name)
    logger.info(config)
    trainpair = [[] for _ in range(20)]
    ########################################################
    FirstTrain = True
    ########################################################
    seed_torch()
    if FirstTrain:
        exp = [[] for _ in range(20)]
        finexp = [[] for _ in range(20)]
        costCache = {}
    else:
        b_file = open('', 'rb')
        exp = pickle.load(b_file)
        modelpath = ''
        b_file.close()
        c_file = open('', 'rb')
        costCache = pickle.load(c_file)
        c_file.close()
        d_file = open('', 'rb')
        finexp = pickle.load(d_file)
        d_file.close()
        getTrainPair(exp, exp, trainpair)
        print('load exp bestsubplans costcache success !!')
    allstime = time.time()
    workload = envs.STACK(envs.STACK.Params())
    workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(workload.workload_info.rel_names)
    # need to change parms
    gamma = 0.25
    learning_rate = 1e-3
    dropbuffer = False

    trainquery, testquery = get_train_test_split(args.experiment)


    dp_Signs = [True for i in range(len(trainquery))]
    sqllist = load_sql_Files(trainquery)
    testsqllist = load_sql_Files(testquery)
    logger.info("Train SQL List {}".format(sqllist))
    sqls = load_sql(sqllist)
    testsqls = load_sql(testsqllist)
    bestplandata = [[[] for _ in range(20)] for _ in range(len(trainquery))]
    bestplanslist = [[] for _ in range(len(sqls))]
    iteration_num = 30

    # initial timeout and it will update in dp
    timeoutlist = setInitialTimeout(sqls, dropbuffer, testtime=3)
    pg_latency_train = getPG_latency(sqls)
    print('pg_base_latency_train', pg_latency_train)
    pg_latency_test = getPG_latency(testsqls)
    print('pg_base_latency_test', pg_latency_test)
    train_gmrl = []
    test_gmrl = []
    logger.info("timeoutList:{}".format(timeoutlist))
    batchsize = 256
    num_inner_epochs = 500
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    maxLevel = 0
    greedy = -1.0
    bestTrainGmrl = 20
    bestTestGmrl = 20
    decay = greedy / (iteration_num * 2)
    # ```````````````
    model_levels = []
    loss_fn = ''
    from util import plans_lib

    nodeFeaturizer = plans_lib.PhysicalTreeNodeFeaturizer(workload.workload_info)
    dpsign = True
    for i in range(0, len(sqls)):
        join_graph, all_join_conds, query_leaves, origin_dp_tables = DP.getPreCondition(sqllist[i])
        dp_tables1 = copy.deepcopy(origin_dp_tables)
        maxLevel = maxLevel if maxLevel > len(query_leaves) else len(query_leaves)
    if not FirstTrain:
        model_levels, optlist = getModelsFromFile(maxLevel, modelpath)
    else:
        model_levels, optlist = getModels(maxLevel)
        
    #
    # Potentially log everything into WandB
    # ================================================================================
    #
    if args.wandb:
        wandb.config['gamma'] = gamma
        wandb.config['learning_rate'] = learning_rate
        wandb.config['dropbuffer'] = dropbuffer
        wandb.config['logs_name'] = logs_name
        wandb.config['train_queries'] = trainquery
        wandb.config['test_queries'] = testquery
        wandb.config['batchsize'] = batchsize
        wandb.config['decay'] = decay
        wandb.config['num_outer_epochs'] = iteration_num
        wandb.config['num_inner_epochs'] = num_inner_epochs
        wandb.config['maxlevel'] = maxLevel

    # for iter in range(0, iteration_num):
    for iter in tqdm(range(iteration_num), desc='Outer Iterations (#iteration_num)...'):
        logger.info('iter {} start!'.format(str(iter)))
        stime = time.time()
        levelList = [{} for _ in range(20)]

        # for i in range(0, len(sqls)):
        for i in tqdm(range(len(sqls)), leave=False, desc=f'Iterating over {len(sqls)} queries...'):
            if dp_Signs[i]:
                join_graph, all_join_conds, query_leaves, origin_dp_tables = DP.getPreCondition(sqllist[i])
                dp_tables1 = copy.deepcopy(origin_dp_tables)
                output1, bestplanhint, num, timeout = DP.dp.UCB_left_prune_replay_fix_kl(join_graph, all_join_conds,
                                                                                         query_leaves,
                                                                                         dp_tables1, workload, exp,
                                                                                         FirstTrain, model_levels,
                                                                                         timeoutlist[i],
                                                                                         dropbuffer=dropbuffer,
                                                                    
                                                                                         nodeFeaturizer=nodeFeaturizer,
                                                                                         greedy=greedy,
                                                                                         subplans_fin=finexp,
                                                                                         finsql=sqls[i],
                                                                                         costCache=costCache,
                                                                                         dpsign=dpsign,
                                                                                         levelList=levelList)
                greedy = greedy - decay
                timeoutlist[i] = round(timeout, 3)
                bestplanslist[i].append([bestplanhint, num])
                getTrainPair(output1, exp, trainpair)
                output1.clear()
            if timeoutlist[i] < pg_latency_train[i] * 0.68:
                dp_Signs[i] = False
        dpsign = not dpsign
        print(dp_Signs)
        logger.info("dptime = {}".format(time.time() - stime))
        logger.info('now timeoutlist = {}'.format(timeoutlist))

        loss_fn = nn.CrossEntropyLoss()

        logger.info('Train start ,iter ={} '.format(iter))
        logger.info(
            'trainpair num ={},now experience num = {},best exp num  = {} '.format(getexpnum(trainpair), getexpnum(exp),
                                                                                   getexpnum(finexp))
        )

        trainTimes = 0
        testTimes = 0
        FirstTrain = False
        for modelnum in tqdm(range(2, len(model_levels)), leave=False, desc='Iterating over models...'):
            optimizer = optlist[modelnum]
            temtrainpair = copy.deepcopy(trainpair[modelnum])
            if len(temtrainpair) < 2:
                continue
            for epoch in tqdm(range(num_inner_epochs), leave=False, desc=f'Training {num_inner_epochs} inner epochs (#num_inner_epochs)...'):
                ttime = time.time()
                shuffled_indices = np.random.permutation(len(temtrainpair))
                # train
                current_idx = 0
                while current_idx < len(shuffled_indices):
                    currentTrainPair = [temtrainpair[idx] for idx in
                                        shuffled_indices[current_idx: current_idx + batchsize]]
                    query_feats = []
                    nodes = []
                    latencies = []
                    costs = []
                    torch.cuda.empty_cache()
                    for i in currentTrainPair:
                        query_feats.append(i[0][0])
                        query_feats.append(i[3][0])
                        nodes.append(i[0][1])
                        nodes.append(i[3][1])
                        latencies.append(i[1])
                        latencies.append(i[4])
                        costs.append(i[2])
                        costs.append(i[5])
                    query_feats = (torch.cat(query_feats, dim=0)).to(DEVICE)
                    trees, indexes = TreeConvFeaturize(nodeFeaturizer, nodes)
                    if torch.cuda.is_available():
                        trees = trees.to(DEVICE)
                        indexes = indexes.to(DEVICE)
                    calibration = []
                    for i in range(10):
                        calibration.append(
                            torch.tanh(model_levels[modelnum](query_feats, trees, indexes).to(DEVICE)).add(1))

                    calibration = torch.cat(calibration, 1)
                    calibration = torch.mean(calibration, dim=1)
                    temloss = calculateLossForBatch(latencies, costs, calibration)
                    loss = torch.mean(temloss, 0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    current_idx += batchsize
                trainTimes = trainTimes + time.time() - ttime
                tetime = time.time()
                acc = 0
                cout = 0
                current_idx = 0
                while current_idx < len(shuffled_indices):

                    currentTrainPair = [temtrainpair[idx] for idx in
                                        shuffled_indices[current_idx: current_idx + batchsize]]
                    query_feats = []
                    nodes = []
                    latencies = []
                    costs = []
                    for i in currentTrainPair:
                        cout = cout + 1
                        query_feats.append(i[0][0])
                        query_feats.append(i[3][0])
                        nodes.append(i[0][1])
                        nodes.append(i[3][1])
                        latencies.append(i[1])
                        latencies.append(i[4])
                        costs.append(i[2])
                        costs.append(i[5])
                    query_feats = (torch.cat(query_feats, dim=0)).to(DEVICE)
                    trees, indexes = TreeConvFeaturize(nodeFeaturizer, nodes)
                    if torch.cuda.is_available():
                        trees = trees.to(DEVICE)
                        indexes = indexes.to(DEVICE)
                    calibration = []
                    for m in range(10):
                        with torch.no_grad():
                            calibration.append(
                                torch.tanh(model_levels[modelnum](query_feats, trees, indexes)).add(1))
                    calibration = torch.cat(calibration, 1)
                    calibration = torch.mean(calibration, dim=1)
                    calibration = calibration.unsqueeze(1)
                    calibration = calibration.view(-1, 2)
                    costs = torch.tensor(costs, device=DEVICE).view(-1, 2)
                    calibratedCost = calibration * costs
                    softm = nn.functional.softmax(calibratedCost, dim=1)
                    prediction = torch.max(softm, dim=1)[1]
                    res = []
                    for i in range(0, len(latencies), 2):
                        if latencies[i] > latencies[i + 1]:
                            res.append(0)
                        else:
                            res.append(1)
                    res = torch.tensor(res, device=DEVICE)
                    current_idx += batchsize
                    acc += torch.sum(res == prediction).data.cpu().numpy().squeeze()
                testTimes = testTimes + time.time() - tetime
                logger.info("iter:{},model:{},train iters:{}, acc:{} ".format(iter, modelnum, epoch + 1, acc / cout))

                if args.wandb:
                    accuracy_key = f'accuracy__model_{modelnum}'
                    inner_epoch_key = f'inner_epoch__model_{modelnum}'
                    wandb.log({
                        accuracy_key: acc/cout,
                        'outer_epoch': iter,
                        inner_epoch_key: epoch+1,
                        'inner_epoch': epoch+1,
                        'combined_epoch': iter*13 + epoch+1
                    })

                if acc / cout > 0.96 or epoch > 13:
                    break

        logger.info('train time ={} test time = {}'.format(trainTimes, testTimes))
        testtime = time.time()

        nowtraingmrl = getGMRL(trainquery, model_levels, pg_latency_train, nodeFeaturizer, costCache, workload,
                               exp=exp, old=pg_latency_train)
        if nowtraingmrl < bestTrainGmrl:
            bestTrainGmrl = nowtraingmrl
            for modelnum in range(2, len(model_levels)):
                modelname = log_dir + '/BestTrainModel_' + logs_name + '_' + str(modelnum) + '.pth'
                torch.save(model_levels[modelnum], modelname)
        train_gmrl.append(nowtraingmrl)
        nowtestgmrl = getGMRL(testquery, model_levels, pg_latency_test, nodeFeaturizer, costCache, workload)
        if nowtestgmrl < bestTestGmrl:
            bestTestGmrl = nowtestgmrl
            for modelnum in range(2, len(model_levels)):
                modelname = log_dir + '/BestTestModel_' + logs_name + '_' + str(modelnum) + '.pth'
                torch.save(model_levels[modelnum], modelname)
        test_gmrl.append(nowtestgmrl)

        logger.info('GMRL test  time ={}'.format(time.time() - testtime))
        logger.info('train_gmrl ={}'.format(train_gmrl))
        logger.info('test_gmrl ={}'.format(test_gmrl))
        
        if args.wandb:
            wandb.log({
                'outer_epoch': iter,
                'gmrl_train': train_gmrl,
                'gmrl_test': test_gmrl,
                'gmrl_train_best': bestTrainGmrl,
                'gmrl_test_best': bestTestGmrl,
            })

        levelList.clear()
        gc.collect()
        a_file = open(log_dir + '/Bestplans_' + logs_name + '.pkl', 'wb')
        b_file = open(log_dir + '/Exp_' + logs_name + '.pkl', 'wb')
        c_file = open(log_dir + '/cost_' + logs_name + '.pkl', 'wb')
        d_file = open(log_dir + '/finexp_' + logs_name + '.pkl', 'wb')
        pickle.dump(exp, b_file)
        pickle.dump(bestplanslist, a_file)
        pickle.dump(costCache, c_file)
        pickle.dump(finexp, d_file)
        a_file.close()
        b_file.close()
        c_file.close()
        d_file.close()
    logger.info('all time = {} '.format(time.time() - allstime))
