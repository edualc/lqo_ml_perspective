from ImportantConfig import Config
from math import e 
from PGUtils import pgrunner
import torch
from KNN import KNN
import time


def formatFloat(t):
    try:
        return " ".join(["{:.4f}".format(x) for x in t])
    except:
        return " ".join(["{:.4f}".format(x) for x in [t]])
config = Config()

class Timer:
    def __init__(self,):
        from time import time
        self.timer = time
        self.startTime = {}
    def reset(self,s):
        self.startTime[s] = self.timer()
    def record(self,s):
        return self.timer()-self.startTime[s]
timer = Timer()
        
class Hinter:
    def __init__(self,model,sql2vec,value_extractor,mcts_searcher=None):
        self.model = model #Net.TreeNet
        self.sql2vec = sql2vec#
        self.value_extractor = value_extractor
        self.pg_planningtime_list = []
        self.pg_runningtime_list = [] #default pg running time
        
        self.mcts_time_list = []#time for mcts
        self.hinter_planningtime_list = [] #chosen hinter running time,include the timeout
        self.MHPE_time_list = []
        self.hinter_runtime_list = []
        
        self.chosen_plan = []#eg((leading ,pg))
        self.hinter_time_list = []#final plan((eg [(leading,),(leading,pg),...]))
        self.knn = KNN(10)
        self.mcts_searcher = mcts_searcher
        self.hinter_times = 0
        
    def findBestHint(self,plan_json_PG,alias,sql_vec,sql):
        alias_id = [self.sql2vec.aliasname2id[a] for a in alias]
        timer.reset('mcts_time_list')
        id_joins_with_predicate = [(self.sql2vec.aliasname2id[p[0]],self.sql2vec.aliasname2id[p[1]]) for p in self.sql2vec.join_list_with_predicate]
        id_joins = [(self.sql2vec.aliasname2id[p[0]],self.sql2vec.aliasname2id[p[1]]) for p in self.sql2vec.join_list]
        leading_length = config.leading_length
        if leading_length==-1:
            leading_length = len(alias)
        if leading_length>len(alias):
            leading_length = len(alias)
        join_list_with_predicate = self.mcts_searcher.findCanHints(40,len(alias),sql_vec,id_joins,id_joins_with_predicate,alias_id,depth=leading_length)
        self.mcts_time_list.append(timer.record('mcts_time_list'))
        
        leading_list = []
        plan_jsons = []
        leadings_utility_list = []
        for join in join_list_with_predicate:
            leading_list.append('/*+Leading('+" ".join([self.sql2vec.id2aliasname[x] for x in join[0][:leading_length]])+')*/')
            leadings_utility_list.append(join[1])
            ##To do: parallel planning
            plan_jsons.append(pgrunner.getCostPlanJson(leading_list[-1]+sql))
        plan_jsons.extend([plan_json_PG])
        timer.reset('MHPE_time_list')
        plan_times = self.predictWithUncertaintyBatch(plan_jsons=plan_jsons,sql_vec = sql_vec)
        self.MHPE_time_list.append(timer.record('MHPE_time_list'))
        chosen_leading_pair = sorted(zip(plan_times[:config.max_hint_num],leading_list,leadings_utility_list),key = lambda x:x[0][0]+self.knn.kNeightboursSample(x[0]))[0]
        return chosen_leading_pair
    
    def hinterRun(self,sql):
        self.hinter_times += 1
        plan_json_PG = pgrunner.getCostPlanJson(sql)
        self.samples_plan_with_time = []
        mask = (torch.rand(1,config.head_num,device = config.device)<0.9).long()
        
        if config.cost_test_for_debug:
            self.pg_runningtime_list.append(pgrunner.getCost(sql)[0])
            self.pg_planningtime_list.append(pgrunner.getCostPlanJson(sql)['Planning Time'])
        else:
            self.pg_runningtime_list.append(pgrunner.getAnalysePlanJson(sql)['Plan']['Actual Total Time'])
            self.pg_planningtime_list.append(pgrunner.getAnalysePlanJson(sql)['Planning Time'])
        
        sql_vec,alias = self.sql2vec.to_vec(sql)
        plan_jsons = [plan_json_PG]
        plan_times = self.predictWithUncertaintyBatch(plan_jsons=plan_jsons,sql_vec = sql_vec)
        
        algorithm_idx = 0
        
        chosen_leading_pair = self.findBestHint(plan_json_PG=plan_json_PG,alias=alias,sql_vec = sql_vec,sql=sql)
        knn_plan = abs(self.knn.kNeightboursSample(plan_times[0]))
        if chosen_leading_pair[0][0]<plan_times[algorithm_idx][0] and abs(knn_plan)<config.threshold and self.value_extractor.decode(plan_times[0][0])>100:
            from math import e
            max_time_out = min(int(self.value_extractor.decode(chosen_leading_pair[0][0])*3),config.max_time_out)
            if config.cost_test_for_debug:
                leading_time_flag = pgrunner.getCost(sql = chosen_leading_pair[1]+sql)
                self.hinter_runtime_list.append(leading_time_flag[0])
                ##To do: parallel planning
                self.hinter_planningtime_list.append(pgrunner.getCostPlanJson(sql = chosen_leading_pair[1]+sql)['Planning Time'])
            else:
                plan_json  = pgrunner.getAnalysePlanJson(sql = chosen_leading_pair[1]+sql)
                leading_time_flag = (plan_json['Plan']['Actual Total Time'],plan_json['timeout'])
                self.hinter_runtime_list.append(leading_time_flag[0])
                ##To do: parallel planning
                self.hinter_planningtime_list.append(plan_json['Planning Time'])
            
            self.knn.insertAValue((chosen_leading_pair[0],self.value_extractor.encode(leading_time_flag[0])-chosen_leading_pair[0][0]))
            if config.cost_test_for_debug:
                self.samples_plan_with_time.append([pgrunner.getCostPlanJson(sql = chosen_leading_pair[1]+sql,timeout=max_time_out),leading_time_flag[0],mask])
            else:
                self.samples_plan_with_time.append([pgrunner.getCostPlanJson(sql = chosen_leading_pair[1]+sql,timeout=max_time_out),leading_time_flag[0],mask])
            if leading_time_flag[1]:
                if config.cost_test_for_debug:
                    pg_time_flag = pgrunner.getCost(sql=sql)
                else:
                    pg_time_flag = pgrunner.getLatency(sql=sql,timeout = 300*1000)
                self.knn.insertAValue((plan_times[0],self.value_extractor.encode(pg_time_flag[0])-plan_times[0][0]))
                if self.samples_plan_with_time[0][1]>pg_time_flag[0]*1.8:
                    self.samples_plan_with_time[0][1] = pg_time_flag[0]*1.8
                    self.samples_plan_with_time.append([plan_json_PG,pg_time_flag[0],mask])
                else:
                    self.samples_plan_with_time[0] = [plan_json_PG,pg_time_flag[0],mask]
                
                self.hinter_time_list.append([max_time_out,pgrunner.getLatency(sql=sql,timeout = 300*1000)[0]])
                self.chosen_plan.append([chosen_leading_pair[1],'PG'])
            else:
                self.hinter_time_list.append([leading_time_flag[0]])
                self.chosen_plan.append([chosen_leading_pair[1]])
        else:
            if config.cost_test_for_debug:
                pg_time_flag = pgrunner.getCost(sql=sql)
                self.hinter_runtime_list.append(pg_time_flag[0])
                ##To do: parallel planning
                self.hinter_planningtime_list.append(pgrunner.getCostPlanJson(sql)['Planning Time'])
            else:
                pg_time_flag = pgrunner.getLatency(sql=sql,timeout = 300*1000)
                self.hinter_runtime_list.append(pg_time_flag[0])
                ##To do: parallel planning

                self.hinter_planningtime_list.append(pgrunner.getAnalysePlanJson(sql = sql)['Planning Time'])
            self.knn.insertAValue((plan_times[0],self.value_extractor.encode(pg_time_flag[0])-plan_times[0][0]))
            self.samples_plan_with_time.append([plan_json_PG,pg_time_flag[0],mask])
            self.hinter_time_list.append([pg_time_flag[0]])
            self.chosen_plan.append(['PG'])

        ## To do: parallel the training process
        ##
        for sample in self.samples_plan_with_time:
            target_value = self.value_extractor.encode(sample[1])
            self.model.train(plan_json = sample[0],sql_vec = sql_vec,target_value=target_value,mask = mask,is_train = True)
            self.mcts_searcher.train(tree_feature = self.model.tree_builder.plan_to_feature_tree(sample[0]),sql_vec = sql_vec,target_value = sample[1],alias_set=alias)
        
        if self.hinter_times<1000 or self.hinter_times%10==0:
            loss=  self.model.optimize()[0]
            loss1 = self.mcts_searcher.optimize()
            if self.hinter_times<1000:
                loss=  self.model.optimize()[0]
                loss1 = self.mcts_searcher.optimize()
            if loss>3:
                loss=  self.model.optimize()[0]
                loss1 = self.mcts_searcher.optimize()
            if loss>3:
                loss=  self.model.optimize()[0]
                loss1 = self.mcts_searcher.optimize()
        

        assert len(set([len(self.hinter_runtime_list),len(self.pg_runningtime_list),len(self.mcts_time_list),len(self.hinter_planningtime_list),len(self.MHPE_time_list),len(self.hinter_runtime_list),len(self.chosen_plan),len(self.hinter_time_list)]))==1
        return self.pg_planningtime_list[-1],self.pg_runningtime_list[-1],self.mcts_time_list[-1],self.hinter_planningtime_list[-1],self.MHPE_time_list[-1],self.hinter_runtime_list[-1],self.chosen_plan[-1],self.hinter_time_list[-1]

    def predictWithUncertaintyBatch(self,plan_jsons,sql_vec):
        sql_feature = self.model.value_network.sql_feature(sql_vec)
        import torchfold
        fold = torchfold.Fold(cuda=True)
        res = []
        multi_list = []
        for plan_json in plan_jsons:
            tree_feature = self.model.tree_builder.plan_to_feature_tree(plan_json)
            multi_value = self.model.plan_to_value_fold(tree_feature=tree_feature,sql_feature = sql_feature,fold=fold)
            multi_list.append(multi_value)
        multi_value = fold.apply(self.model.value_network,[multi_list])[0]
        mean,variance  = self.model.mean_and_variance(multi_value=multi_value[:,:config.head_num])
        v2 = torch.exp(multi_value[:,config.head_num]*config.var_weight).data.reshape(-1)
        if isinstance(mean,float):
            mean_item = [mean]
        else:
            mean_item = [x.item()for x in mean]
        if isinstance(variance,float):
            variance_item = [variance]
        else:
            variance_item = [x.item()for x in variance]
        # variance_item = [x.item() for x in variance]
        if isinstance(v2,float):
            v2_item = [v2]
        else:
            v2_item = [x.item()for x in v2]
        # v2_item = [x.item() for x in v2]
        res = list(zip(mean_item,variance_item,v2_item))
        return res