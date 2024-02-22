import sys
sys.path.append(".")

from JOBParser import TargetTable,FromTable,Comparison
# max_column_in_table = 15
import torch
import torch
import torch.nn as nn
from itertools import count
import numpy as np
from PGUtils import pgrunner
from JOBParser import TargetTable,FromTable,Comparison
from ImportantConfig import Config
config = Config()
def zero_hc(input_dim = 1):
    return torch.zeros(input_dim,config.hidden_size,device = config.device),torch.zeros(input_dim,config.hidden_size,device = config.device)
column_id = {}
def getColumnId(column):
    if not column in column_id:
        column_id[column] = len(column_id)
    return column_id[column]
class Sql2Vec:
    def __init__(self,):
        pass
    def to_vec(self,sql):
        from psqlparse import parse_dict
        self.sql = sql
        import time
        # startTime = time.time()
        parse_result = parse_dict(self.sql)[0]["SelectStmt"]
        self.target_table_list = [TargetTable(x["ResTarget"]) for x in parse_result["targetList"]]
        self.from_table_list = [FromTable(x["RangeVar"]) for x in parse_result["fromClause"]]
        if len(self.from_table_list)<2:
            return
        self.aliasname2fullname = {}

        self.id2aliasname = config.id2aliasname
        self.aliasname2id = config.aliasname2id
        # self.id2aliasname = {0: 'start', 1: 'chn', 2: 'ci', 3: 'cn', 4: 'ct', 5: 'mc', 6: 'rt', 7: 't', 8: 'k', 9: 'lt', 10: 'mk', 11: 'ml', 12: 'it1', 13: 'it2', 14: 'mi', 15: 'mi_idx', 16: 'it', 17: 'kt', 18: 'miidx', 19: 'at', 20: 'an', 21: 'n', 22: 'cc', 23: 'cct1', 24: 'cct2', 25: 'it3', 26: 'pi', 27: 't1', 28: 't2', 29: 'cn1', 30: 'cn2', 31: 'kt1', 32: 'kt2', 33: 'mc1', 34: 'mc2', 35: 'mi_idx1', 36: 'mi_idx2', 37: 'an1', 38: 'n1', 39: 'a1'}
        # self.aliasname2id = {'kt1': 31, 'chn': 1, 'cn1': 29, 'mi_idx2': 36, 'cct1': 23, 'n': 21, 'a1': 39, 'kt2': 32, 'miidx': 18, 'it': 16, 'mi_idx1': 35, 'kt': 17, 'lt': 9, 'ci': 2, 't': 7, 'k': 8, 'start': 0, 'ml': 11, 'ct': 4, 't2': 28, 'rt': 6, 'it2': 13, 'an1': 37, 'at': 19, 'mc2': 34, 'pi': 26, 'mc': 5, 'mi_idx': 15, 'n1': 38, 'cn2': 30, 'mi': 14, 'it1': 12, 'cc': 22, 'cct2': 24, 'an': 20, 'mk': 10, 'cn': 3, 'it3': 25, 't1': 27, 'mc1': 33}
        
        self.join_list = set()
        self.aliasnames_root_set = set([x.getAliasName() for x in self.from_table_list])

        self.alias_selectivity = np.asarray([0]*len(self.id2aliasname),dtype = float)
        self.aliasname2fromtable = {}
        for table in self.from_table_list:
            self.aliasname2fromtable[table.getAliasName()] = table
            self.aliasname2fullname[table.getAliasName()] = table.getFullName()
        
        self.aliasnames = set(self.aliasname2fromtable.keys())
        self.comparison_list =[Comparison(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]
        self.total = 0
        self.join_matrix = np.zeros((len(self.id2aliasname),len(self.id2aliasname)),dtype = float)
        self.count_selectivity = np.asarray([0]*config.max_column,dtype = float)
        self.has_predicate = set()
        self.join_list_with_predicate = set()
        for comparison in self.comparison_list:
            if len(comparison.aliasname_list) == 2:
                left_aliasname = comparison.aliasname_list[0]
                right_aliasname = comparison.aliasname_list[1]
                idx0 = self.aliasname2id[left_aliasname]
                idx1 = self.aliasname2id[right_aliasname]
                if idx0<idx1:
                    self.join_list.add((left_aliasname,right_aliasname))
                else:
                    self.join_list.add((right_aliasname,left_aliasname))
                self.join_matrix[idx0][idx1] = 1
                self.join_matrix[idx1][idx0] = 1
            else:
                left_aliasname = comparison.aliasname_list[0]
                # self.alias_selectivity[self.aliasname2id[left_aliasname]] = max(self.alias_selectivity[self.aliasname2id[left_aliasname]],pgrunner.getSelectivity(str(self.aliasname2fromtable[comparison.aliasname_list[0]]),str(comparison)))
                self.alias_selectivity[self.aliasname2id[left_aliasname]] = self.alias_selectivity[self.aliasname2id[left_aliasname]]+pgrunner.getSelectivity(str(self.aliasname2fromtable[comparison.aliasname_list[0]]),str(comparison))
                self.has_predicate.add(left_aliasname)
                self.count_selectivity[getColumnId(comparison.column)] = self.count_selectivity[getColumnId(comparison.column)]+pgrunner.getSelectivity(str(self.aliasname2fromtable[comparison.aliasname_list[0]]),str(comparison))
        for ajoin in self.join_list:
            if ajoin[0] in self.has_predicate or ajoin[1] in self.has_predicate :
                self.join_list_with_predicate.add(ajoin)
        if config.max_column==40:
            return np.concatenate((self.join_matrix.flatten(),self.alias_selectivity)), self.aliasnames_root_set
        # print(np.concatenate((self.join_matrix.flatten(),self.count_selectivity)).shape)
        return np.concatenate((self.join_matrix.flatten(),self.count_selectivity)), self.aliasnames_root_set

JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]
LEAF_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Index Scan"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES

class ValueExtractor:
    def __init__(self,offset=config.offset,max_value = 20):
        self.offset = offset
        self.max_value = max_value
    # def encode(self,v):
    #     return np.log(self.offset+v)/np.log(2)/self.max_value
    # def decode(self,v):
    #     # v=-(v*v<0)
    #     return np.exp(v*self.max_value*np.log(2))#-self.offset
    def encode(self,v):
        return int(np.log(2+v)/np.log(config.max_time_out)*200)/200.
        return int(np.log(self.offset+v)/np.log(config.max_time_out)*200)/200.
    def decode(self,v):
        # v=-(v*v<0)
        # return np.exp(v/2*np.log(config.max_time_out))#-self.offset
        return np.exp(v*np.log(config.max_time_out))#-self.offset
    def cost_encode(self,v,min_cost,max_cost):
        return (v-min_cost)/(max_cost-min_cost)
    def cost_decode(self,v,min_cost,max_cost):
        return (max_cost-min_cost)*v+min_cost
    def latency_encode(self,v,min_latency,max_latency):
        return (v-min_latency)/(max_latency-min_latency)
    def latency_decode(self,v,min_latency,max_latency):
        return (max_latency-min_latency)*v+min_latency
    def rows_encode(self,v,min_cost,max_cost):
        return (v-min_cost)/(max_cost-min_cost)
    def rows_decode(self,v,min_cost,max_cost):
        return (max_cost-min_cost)*v+min_cost
value_extractor = ValueExtractor()
def get_plan_stats(data):
    return [value_extractor.encode(data["Total Cost"]),value_extractor.encode(data["Plan Rows"])]

class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg

def is_join(node):
    return node["Node Type"] in JOIN_TYPES

def is_scan(node):
    return node["Node Type"] in LEAF_TYPES

# fasttext
class PredicateEncode:
    def __init__(self,):
        pass
    def stringEncoder(self,string_predicate):
        return torch.tensor([0,1]+[0]*config.hidden_size,device = config.device).float()
        pass
    def floatEncoder(self,float1,float2):
        return torch.tensor([float1,float2]+[0]*config.hidden_size,device = config.device).float()
        pass
class TreeBuilder:
    def __init__(self):
        self.__stats = get_plan_stats
        self.id2aliasname = config.id2aliasname
        self.aliasname2id = config.aliasname2id
        
    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.__relations:
                if rel in node[name_key]:
                    return rel

            raise TreeBuilderError("Could not find relation name for bitmap index scan")

        raise TreeBuilderError("Cannot extract relation type from node")
    def __alias_name(self, node):
        if "Alias" in node:
            return np.asarray([self.aliasname2id[node["Alias"]]])

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Cond" #if "Index Cond" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.aliasname2id:
                if rel+'.' in node[name_key]:
                    return np.asarray([-1])
                    return np.asarray([self.aliasname2id[rel]])

        #     raise TreeBuilderError("Could not find relation name for bitmap index scan")
        print(node)
        raise TreeBuilderError("Cannot extract Alias type from node")
                
    def __featurize_join(self, node):
        assert is_join(node)
        # return [node["Node Type"],self.__stats(node),0,0]
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr, self.__stats(node)))
        feature = torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(-1,config.input_size)
        return feature

    def __featurize_scan(self, node):
        assert is_scan(node)
        # return [node["Node Type"],self.__stats(node),self.__alias_name(node)]
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        feature = np.concatenate((arr, self.__stats(node)))
        feature = torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(-1,config.input_size)
        return (feature,
                torch.tensor(self.__alias_name(node),device = config.device,dtype = torch.long))

    def plan_to_feature_tree(self, plan):
        
        
        # children = plan["Plans"] if "Plans" in plan else []
        if "Plan" in plan:
            plan = plan["Plan"]
        children = plan["Plan"] if "Plan" in plan else (plan["Plans"] if "Plans" in plan else [])
        if len(children) == 1:
            child_value = self.plan_to_feature_tree(children[0])
            if "Alias" in plan and plan["Node Type"]=='Bitmap Heap Scan':
                alias_idx_np = np.asarray([self.aliasname2id[plan["Alias"]]])
                if isinstance(child_value[1],tuple):
                    raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))
                return (child_value[0],torch.tensor(alias_idx_np,device = config.device,dtype = torch.long))
            return child_value
        # print(plan)
        if is_join(plan):
            assert len(children) == 2
            my_vec = self.__featurize_join(plan)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            # print('is_join',my_vec)
            return (my_vec, left, right)

        if is_scan(plan):
            assert not children
            # print(plan)
            s = self.__featurize_scan(plan)
            # print('is_scan',s)
            return s

        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))


        
                
                
