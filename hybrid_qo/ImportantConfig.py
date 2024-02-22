import torch
from math import log
class Config:
    def __init__(self,):
        # self.datafile = 'JOBqueries.workload'
        self.schemaFile = "schema.sql"
        self.user = 'postgres'
        self.password = 'postgres'
        self.dataset = 'JOB'
        self.userName = self.user
        self.usegpu = True
        self.head_num = 10
        self.input_size = 9 # 7+2, 7 = All types (scans/joins), 2= ['total cost', 'plan rows']
        self.hidden_size = 64
        self.batch_size = 256
        self.ip = "pg_balsa"
        self.port = 5432
        self.device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.cpudevice = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
        self.var_weight = 0.00 #for au, 0:disable,0.01:enable
        self.cost_test_for_debug = False
        self.max_hint_num = 20 # TopN?
        self.max_time_out = 3 * 60 * 1000
        self.threshold = log(3)/log(self.max_time_out)
        self.leading_length = 2
        self.try_hint_num = 3
        self.mem_size = 2000
        self.mcts_v = 1.1
        self.searchFactor = 4
        self.U_factor = 0.0
        self.log_file = 'log_c3_h64_s4_t3.txt'
        self.latency_file = 'latency_record.txt'
        self.modelpath = 'model/'
        self.offset = 20 # Offset for numerical stability?
        
        # JOB
        # -------------------------------------
        # self.database = 'imdbload'
        # self.max_alias_num  = 40
        # self.id2aliasname = {0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt', 10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt', 18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3', 26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1', 34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'}
        # self.aliasname2id = {'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32, 'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8, 'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19, 'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12, 'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33}
        # self.max_column = 100 # Number of columns in the database (JOB=108, but not all are used in queries?)

        # self.n_epochs = 1
        # self.queries_file = 'workload/JOB_static.json'

        # self.n_epochs = 50
        # self.queries_file = 'workload/JOB__base_query_split_1__train.json'
        # self.queries_file = 'workload/JOB__base_query_split_2__train.json'
        # self.queries_file = 'workload/JOB__base_query_split_3__train.json'
        # self.queries_file = 'workload/JOB__leave_one_out_split_1__train.json'
        # self.queries_file = 'workload/JOB__leave_one_out_split_2__train.json'
        # self.queries_file = 'workload/JOB__leave_one_out_split_3__train.json'
        # self.queries_file = 'workload/JOB__random_split_1__train.json'
        # self.queries_file = 'workload/JOB__random_split_2__train.json'
        # self.queries_file = 'workload/JOB__random_split_3__train.json'

        # STACK
        # -------------------------------------
        self.database = 'so'
        self.n_epochs = 50
        self.max_alias_num  = 29
        self.id2aliasname = { 0: 'start', 1: 'a1', 2: 'acc', 3: 'account', 4: 'b', 5: 'b1', 6: 'b2', 7: 'c1', 8: 'c2', 9: 'pl', 10: 'q', 11: 'q1', 12: 'q2', 13: 'question', 14: 's', 15: 's1', 16: 's2', 17: 'site', 18: 'so_user', 19: 't', 20: 't1', 21: 't2', 22: 'tag', 23: 'tag_question', 24: 'tq', 25: 'tq1', 26: 'tq2', 27: 'u1', 28: 'u2' }
        self.aliasname2id = { 'start': 0, 'a1': 1, 'acc': 2, 'account': 3, 'b': 4, 'b1': 5, 'b2': 6, 'c1': 7, 'c2': 8, 'pl': 9, 'q': 10, 'q1': 11, 'q2': 12, 'question': 13, 's': 14, 's1': 15, 's2': 16, 'site': 17, 'so_user': 18, 't': 19, 't1': 20, 't2': 21, 'tag': 22, 'tag_question': 23, 'tq': 24, 'tq1': 25, 'tq2': 26, 'u1': 27, 'u2': 28 }
        self.max_column = 66 # Number of columns in the database (JOB=108, but not all are used in queries?)

        self.queries_file = 'workload/STACK__base_query_split_1__train.json'
        # self.queries_file = 'workload/STACK__leave_one_out_split_1__train.json'
        # self.queries_file = 'workload/STACK__random_split_1__train.json'


        self.mcts_input_size = self.max_alias_num*self.max_alias_num+self.max_column