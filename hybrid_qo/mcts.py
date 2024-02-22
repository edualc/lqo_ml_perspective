from __future__ import division
import time
import math
import random
from copy import deepcopy,copy
from tracemalloc import start
import numpy as np
from ImportantConfig import Config
config = Config()

# from models import ValueNet
import torch
# model_path = './saved_models/supervised.pt'
from NET import ValueNet
predictionNet = ValueNet(config.mcts_input_size, n_words=config.max_alias_num, hidden_size=config.hidden_size).to(config.cpudevice)
for name, param in predictionNet.named_parameters():
    from torch.nn import init
    # print(name,param.shape)
    if len(param.shape)==2:
        init.xavier_normal_(param)
    else:
        init.uniform_(param)
# predictionNet = ValueNet(856, 5)
# predictionNet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
# predictionNet.eval()
import torch.nn.functional as F
import torch.optim as optim
import torch
optimizer = optim.Adam(predictionNet.parameters(),lr = 3e-4   ,betas=(0.9,0.999))
# optimizer = optim.SGD(predictionNet.parameters(),lr = 3e-5 )
loss_function = F.smooth_l1_loss

def getValue(inputState1,inputState2):
    with torch.no_grad():
        predictionRuntime = predictionNet(inputState1,inputState2)
    prediction = predictionRuntime.detach().cpu().numpy()[0]/10
    return prediction
import time
def getReward(state):
    inputState1 = state.inputState1
    # inputState1 = torch.tensor(state.queryEncode, dtype=torch.float32).to(config.cpudevice)
    inputState2 = torch.tensor(state.order_list, dtype=torch.long).to(config.cpudevice)
    startTime = time.time()
    with torch.no_grad():
        predictionRuntime = predictionNet(inputState1,inputState2)
    prediction = predictionRuntime.detach().cpu().numpy()[0][0]/10
    # print('prediction',prediction)
    return prediction,time.time()-startTime


from math import log
def flog(x):
    return int((log((x+config.offset)/config.max_time_out)/log(config.mcts_v)))/int((log(config.offset/config.max_time_out)/log(config.mcts_v)))
def eflog(x):
    x = x*int((log(config.offset/config.max_time_out)/log(config.mcts_v)))*log(config.mcts_v)
    from math import e
    return e**x*config.max_time_out
def randomPolicy(node):
    import time
    t1 = 0
    while not node.isTerminal:
        # print(node.state.currentStep)
        # print(node.state.order_list)
        # print(node.state.joins)
        startTime = time.time()
        try:
            temp = node.state.getPossibleActions()
            action = random.choice(list(temp))
        except IndexError:
            raise Exception("Non-terminal state has no possible actions: " + str(state))
        newNode = treeNode(node.state.takeAction(action), node)
        node.children[action] = newNode
        if len(node.state.getPossibleActions()) == len(node.children):
            node.isFullyExpanded = True
        node = newNode
    # reward = state.getReward()
    startTime = time.time()
    reward,nntime = getReward(node.state)
    t1+= time.time()-startTime
    # print(reward)
    return node,reward,t1

getPossibleActionsTime = 0
takeActionTime = 0
import time
class planState:
    def __init__(self, totalNumberOfTables, numberOfTables, queryEncode,all_joins,joins_with_predicate,nodes):
        """[summary]

        Args:
            totalNumberOfTables ([type]): [description]
            numberOfTables ([type]): [description]
            queryEncode ([type]): [description]
            all_joins ([list of int]): 
            nodes ([type]): [description]
        """
        self.tableNumber = totalNumberOfTables
        self.inputState1 = torch.tensor(queryEncode, dtype=torch.float32).to(config.cpudevice)
        self.currentStep = 0
        self.numberOfTables = numberOfTables
        self.queryEncode = queryEncode
        self.order_list = np.zeros(config.max_hint_num,dtype = int)
        self.joins = []
        self.joins_with_predicate = []
        # print("all_joins",len(all_joins))
        self.join_matrix = {}
        for p in all_joins:
            self.join_matrix[p[0]]=set()
            self.join_matrix[p[1]]=set()
            if p[0]<p[1]:
                self.joins.append((p[0],p[1]))
            else:
                self.joins.append((p[1],p[0]))
        for p in joins_with_predicate:
            if p[0]<p[1]:
                self.joins_with_predicate.append((p[0],p[1]))
            else:
                self.joins_with_predicate.append((p[1],p[0]))
        for p in all_joins:
            self.join_matrix[p[0]].add(p[1])
            self.join_matrix[p[1]].add(p[0])
        self.nodes = nodes
        self.possibleActions = []
        
    def getPossibleActions(self):
        global getPossibleActionsTime
        startTime = time.time()
        # print(self.nodes)
        if len(self.possibleActions)>0 and self.currentStep>1:
            return self.possibleActions
        
        possibleActions = set()
        if self.currentStep == 1:
            for join in self.joins_with_predicate:
                if join[0] == self.order_list[0]:
                    possibleActions.add(join[1])
        elif self.currentStep == 0:
            for join in self.joins_with_predicate:
                possibleActions.add(join[0])
        else:
            order_list_set = list(self.order_list)
            for join in self.joins:
                if   join[0] in order_list_set and (not join[1] in order_list_set):
                    possibleActions.add(join[1])
                elif join[1] in order_list_set and (not join[0] in order_list_set):
                    possibleActions.add(join[0])
        
        # possibleActions = list(possibleActions)
        self.possibleActions = possibleActions
        getPossibleActionsTime += time.time()-startTime
        return possibleActions

    def takeAction(self, action):
        global takeActionTime
        startTime = time.time()
        newState = copy(self)
        newState.order_list = copy(self.order_list)
        newState.possibleActions = copy(self.possibleActions)
        # if self.currentStep>0:
        #     print(self.currentStep)
        #     print(self.order_list)
        #     print(self.possibleActions)
        #     newState.order_list[0]=-1
        #     newState.currentStep = -1
        #     newState.possibleActions.add(-1)
        #     print(self.currentStep)
        #     print(self.order_list)
        #     print(self.possibleActions)
        #     print(newState.currentStep)
        #     print(newState.order_list)
        #     print(newState.possibleActions)
        #     raise 
        newState.order_list[newState.currentStep] = action
        newState.currentStep = self.currentStep + 1
        # newState.possibleActions = set()
        newState.possibleActions.remove(action)
        order_list = newState.order_list
        for p in newState.join_matrix[action]:
            if not p in order_list:
                newState.possibleActions.add(p)
        takeActionTime += time.time()-startTime
        return newState

    def isTerminal(self):
        if self.currentStep == self.numberOfTables:
            return True
        return False


class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.children = {}


from collections import namedtuple
MCTSSample = namedtuple('MCTSSample',
                    ('sql_feature', 'order_feature','label'))
class MCTSMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        data =  MCTSSample(*args)
        position = self.position
        self.memory[position] = data
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        if len(self.memory)>batch_size:
            import random
            return random.sample(self.memory, batch_size)
        else:
            return self.memory
    def __len__(self):
        return len(self.memory)
    def resetMemory(self,):
        self.memory =[]
        self.position = 0

class mcts():
    def __init__(self, iterationLimit=None, explorationConstant=1 / math.sqrt(16),
                 rolloutPolicy=randomPolicy):
        if iterationLimit == None:
            raise ValueError("Must have either a time limit or an iteration limit")
        # number of iterations of the search
        if iterationLimit < 1:
            raise ValueError("Iteration limit must be greater than one")
        self.searchLimit = iterationLimit
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy
        self.nntime = 0
        self.nntime_no_feature =0
        global getPossibleActionsTime
        getPossibleActionsTime = 0
        global takeActionTime
        takeActionTime = 0

    def search(self, initialState):
        self.root = treeNode(initialState, None)
        for i in range(self.searchLimit):
            self.executeRound()

        # bestChild = self.getBestChild(self.root, 0)
        # return self.getAction(self.root, bestChild)
    def continueSearch(self):
        for i in range(self.searchLimit):
            self.executeRound()

    def executeRound(self):
        node = self.selectNode(self.root)
        # newState = deepcopy(node.state)
        startTime = time.time()
        node,reward,nntime_no_feature = self.rollout(node)
        self.nntime += time.time()-startTime
        self.nntime_no_feature += nntime_no_feature
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children:
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                # if newNode.isTerminal:
                #     print(newNode)
                return newNode
        print(len(actions),len(node.children))
        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        # print(reward)
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            nodeValue = child.totalReward / child.numVisits + explorationValue * math.sqrt(
                2 * math.log(node.numVisits) / child.numVisits)
            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action


class MCTSHinterSearch():
    def __init__(self,m_size=5000):
        self.memory = MCTSMemory(m_size)
        self.Utility=[]
        self.total_cnt = 0
        self.modelhead = config.log_file.split("/")[-1].split(".txt")[0]
    def dfs(self,node,depth):
        if (node.state.currentStep==depth):
            nodeValue = node.totalReward / node.numVisits
            self.Utility.append((node.state.order_list, eflog(nodeValue)))
            return
        for child in node.children.values():
            self.dfs(child,depth)
    def savemodel(self,):
        torch.save(predictionNet.cpu().state_dict(), 'model/'+self.modelhead+".pth")
        predictionNet.cuda()
    def loadmodel(self,):
        predictionNet.load_state_dict(torch.load('model/'+self.modelhead+".pth"))
        predictionNet.cuda()
    def findCanHints(self, totalNumberOfTables, numberOfTables, queryEncode,all_joins,joins_with_predicate,nodes,depth=2):
        self.total_cnt +=1
        if self.total_cnt%200==0:
            self.savemodel()
        initialState = planState(totalNumberOfTables, numberOfTables, queryEncode,
                                all_joins,joins_with_predicate,nodes)
        
        searchFactor = config.searchFactor
        currentState = initialState
        # print(len(currentState.getPossibleActions()))
        self.mct = mcts(iterationLimit=(int)(len(currentState.getPossibleActions()) *  searchFactor)) 
        self.Utility = []
        self.mct.search(initialState = currentState)
        self.dfs(self.mct.root, depth)
        benefit_top_hints = sorted(self.Utility,key = lambda x :x[1],reverse=True)
        # print("-----start benefit------")
        # for x in benefit_top_hints[:2]:
        #     print(x[0][:10],eflog(x[1]),x[1])
        global getPossibleActionsTime #debug
        self.getPossibleActionsTime = getPossibleActionsTime
        global takeActionTime #debug
        self.takeActionTime = takeActionTime
        return benefit_top_hints[:config.try_hint_num]
    def loss(self,input,target,optimize = True):
        # print(input,target)
        loss_value = loss_function(input=input, target=target)
        if not optimize:
            return loss_value.item()
        optimizer.zero_grad()
        loss_value.backward()
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-0.5*10, 0.5*10)
        optimizer.step()
        return loss_value.item()    
    
    def train(self,tree_feature,sql_vec,target_value,alias_set,is_train=False):
        def plan_to_count(tree_feature):
    #     alias_list = []
            def recursive(tree_feature):
                if isinstance(tree_feature[1],tuple):
                    feature = tree_feature[0]
                    alias_list0 = recursive(tree_feature=tree_feature[1])
                    alias_list1 = recursive(tree_feature=tree_feature[2])
                    if len(alias_list1)==1:
                        return alias_list0+alias_list1
                    if len(alias_list0)==1:
                        return alias_list1+alias_list0
                    return []
                else:
                    return [tree_feature[1].item()]
            return recursive(tree_feature=tree_feature)
        tree_alias = plan_to_count(tree_feature)
        import json
        if len(tree_alias)!=len(alias_set):
            return
        if tree_alias[0]>tree_alias[1]:
            tmp = tree_alias[0]
            tree_alias[0] = tree_alias[1]
            tree_alias[1] = tmp
        tree_alias = tree_alias + [0] * (config.max_hint_num-len(tree_alias))
        inputState1 = torch.tensor(sql_vec, dtype=torch.float32).to(config.cpudevice)
        inputState2 = torch.tensor(tree_alias, dtype=torch.long).to(config.cpudevice)
        predictionRuntime = predictionNet(inputState1,inputState2)
        if target_value>config.max_time_out:
            target_value = config.max_time_out
        label = torch.tensor([(flog(target_value))*10],device = config.cpudevice,dtype = torch.float32)
        # print('label',label)
        loss_value = self.loss(input=predictionRuntime,target=label,optimize=True)
        
        
        
        self.addASample(inputState1,inputState2,label)
        
        return loss_value
    
    
    def optimize(self):
        samples = self.memory.sample(config.batch_size)
        sql_features = []
        order_features = []
        labels = []
        if len(samples)==0:
            return
        for one_sample in samples:
            sql_features.append(one_sample.sql_feature)
            order_features.append(one_sample.order_feature)
            labels.append(one_sample.label)
        sql_feature = torch.stack(sql_features).to(config.cpudevice)
        order_feature = torch.stack(order_features).to(config.cpudevice)
        label = torch.stack(labels,dim = 0).reshape(-1,1)
        predictionRuntime = predictionNet(sql_feature,order_feature)
        loss_value = self.loss(input=predictionRuntime,target=label,optimize=True)
        return loss_value
    def addASample(self,sql_feature,order_feature,label):
        """[summary]

        Args:
            sql_feature ([type]): tensor
            order_feature ([type]): tensor
            ptime ([type]): time in seconds
        """
        self.memory.push(sql_feature,order_feature,label)
