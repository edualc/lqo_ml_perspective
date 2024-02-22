import psycopg2
import json
from math import log
from ImportantConfig import Config

TIMEOUT_MS = 3 * 60 * 1000
NUM_EXECUTIONS = 3

latency_record_dict = {}
# selectivity_dict = {}
latency_record_file = None
config  = Config()

class PGConfig:
    def __init__(self):
        self.keepExecutedPlan =True
        self.maxTimes = NUM_EXECUTIONS
        self.maxTime = TIMEOUT_MS

class PGGRunner:
    def __init__(self,dbname = '',user = '',password = '',host = '',port = '',need_latency_record = True,latency_file = "RecordFile.json"):
        """
        :param dbname:
        :param user:
        :param password:
        :param host:
        :param port:
        :param latencyRecord:-1:loadFromFile
        :param latencyRecordFile:
        """
        self.con = psycopg2.connect(database=dbname, user=user,
                               password=password, host=host, port=port)
        self.cur = self.con.cursor()
        self.config = PGConfig()
        self.need_latency_record = need_latency_record

        self.cur.execute("load 'pg_hint_plan';")
        global latency_record_file
        self.cost_plan_json = {}
        if need_latency_record:
            latency_record_file = self.generateLatencyPool(latency_file)


    def generateLatencyPool(self,fileName):
        """
        :param fileName:
        :return:
        """
        import os
        import json
        # print('in generateLatencyPool')
        
        if os.path.exists(fileName):
            f = open(fileName,"r")
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                global latency_record_dict
                if data[0].find('/*+Leading')==-1:
                    if not data[0] in latency_record_dict:
                        latency_record_dict[data[0]] = data[1]
            f = open(fileName,"a")
        else:
            f = open(fileName,"w")
        return f
    def addLatency(self,k,v):
        latency_record_dict[k] =  v
        latency_record_file.write(json.dumps([k,v])+"\n")
        latency_record_file.flush()
    
    def getAnalysePlanJson(self, sql, timeout=TIMEOUT_MS):
        if config.cost_test_for_debug:
            raise
        
        # MANUALLY DISABLED QUERY CACHING
        # if sql in latency_record_dict:
        #     return latency_record_dict[sql]
        
        try:
            # Changed PG configuration to disable geqo if the join order is hinted
            if 'Leading' in sql:
                self.cur.execute("SET geqo TO off;")
            else:
                self.cur.execute("SET geqo TO on;")
                self.cur.execute("SET geqo_threshold = 12;")
            self.cur.execute("SET statement_timeout = "+str(timeout)+ ";")

            for _ in range(NUM_EXECUTIONS):
                self.cur.execute("explain (COSTS, FORMAT JSON, ANALYSE) "+sql)
                rows = self.cur.fetchall()
            
            plan_json = rows[0][0][0]
            plan_json['timeout'] = False
        except KeyboardInterrupt:
            raise
        except:
            plan_json = {}
            plan_json['Planning Time'] = 20
            plan_json['Plan'] = {'Actual Total Time':config.max_time_out}
            plan_json['timeout'] = True
            self.con.commit()
        if not plan_json['timeout']:
            self.addLatency(sql,plan_json)
        return plan_json
        
    
    def getLatency(self,sql,timeout = TIMEOUT_MS):
        """
        :param sql:a sqlSample object.
        
        :return: the latency of sql
        """
        if config.cost_test_for_debug:
            raise
        plan_json = self.getAnalysePlanJson(sql,timeout)
        
        latency = plan_json['Plan']['Actual Total Time']
        timeout = plan_json['timeout']
        # print(f"getLatency: {latency:.4f} (Timeout: {timeout})")
        return latency, timeout

    
    def getAnalysePlanJsonNoCache(self,sql,timeout=TIMEOUT_MS):
        if config.cost_test_for_debug:
            raise
        
        try:
            # Changed PG configuration to disable geqo if the join order is hinted
            if 'Leading' in sql:
                self.cur.execute("SET geqo TO off;")
            else:
                self.cur.execute("SET geqo TO on;")
                self.cur.execute("SET geqo_threshold = 12;")
            self.cur.execute("SET statement_timeout = "+str(timeout)+ ";")

            for _ in range(NUM_EXECUTIONS):
                self.cur.execute("explain (COSTS, FORMAT JSON, ANALYSE) "+sql)
                rows = self.cur.fetchall()
            
            plan_json = rows[0][0][0]
            plan_json['timeout'] = False
        except KeyboardInterrupt:
            raise
        except:
            plan_json = {}
            plan_json['Plan'] = {'Actual Total Time':config.max_time_out}
            plan_json['timeout'] = True
            self.con.commit()
        return plan_json
        
    
    def getLatencyNoCache(self,sql,timeout = TIMEOUT_MS):
        """
        :param sql:a sqlSample object.
        
        :return: the latency of sql
        """
        if config.cost_test_for_debug:
            raise
        plan_json = self.getAnalysePlanJsonNoCache(sql,timeout)
        return plan_json['Plan']['Actual Total Time'],plan_json['timeout']

    def getResult(self, sql):
        """
        :param sql:a sqlSample object
        :return: the latency of sql
        """
        self.cur.execute("SET statement_timeout = 300000;")
        import time
        self.cur.execute(sql)
        rows = self.cur.fetchall()
        return rows
    def getCostPlanJson(self,sql,timeout=TIMEOUT_MS):
        if sql in self.cost_plan_json:
            return self.cost_plan_json[sql]
        import time
        startTime = time.time()
        # Changed PG configuration to disable geqo if the join order is hinted
        if 'Leading' in sql:
            self.cur.execute("SET geqo TO off;")
        else:
            self.cur.execute("SET geqo TO on;")
            self.cur.execute("SET geqo_threshold = 12;")
        self.cur.execute("SET statement_timeout = "+str(timeout)+ ";")
        self.cur.execute("explain (COSTS, FORMAT JSON) "+sql)
        rows = self.cur.fetchall()
        plan_json = rows[0][0][0]
        plan_json['Planning Time'] = time.time()-startTime
        self.cost_plan_json[sql] = plan_json
        return plan_json
        
    def getCost(self,sql):
        """
        :param sql: a sqlSample object
        :return: the cost of sql
        """
        plan_json = self.getCostPlanJson(sql)
        return plan_json['Plan']['Total Cost'],0
        
    def getSelectivity(self,table,whereCondition):
        global latency_record_dict
        if whereCondition in latency_record_dict:
            return latency_record_dict[whereCondition]
        self.cur.execute("SET statement_timeout = "+str(int(100000))+ ";")
        totalQuery = "select * from "+table+";"
        self.cur.execute("EXPLAIN "+totalQuery)
        rows = self.cur.fetchall()[0][0]
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from "+table+" Where "+whereCondition+";"
        self.cur.execute("EXPLAIN  "+resQuery)
        rows = self.cur.fetchall()[0][0]
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        latency_record_dict[whereCondition] = -log(select_rows/total_rows)
        self.addLatency(whereCondition,-log(select_rows/total_rows))
        return latency_record_dict[whereCondition]

from itertools import count
from pathlib import Path
pgrunner = PGGRunner(config.database,config.user,config.password,config.ip,config.port,need_latency_record=True,latency_file=config.latency_file)