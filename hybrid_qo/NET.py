import torch
import torch.nn.functional as F
import torch.optim as optim
import torchfold
import numpy as np
import torch.nn as nn
from collections import namedtuple
from ImportantConfig import Config
from TreeLSTM import MSEVAR
config = Config()
Transition = namedtuple('Transition',
                    ('tree_feature', 'sql_feature', 'target_feature', 'mask','weight'))
import random
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        data =  Transition(*args)
        position = self.position
        self.memory[position] = data
        self.position = (self.position + 1) % self.capacity
    def weight_sample(self,batch_size):
        import random
        weight = []
        current_weight = 0
        for x in self.memory:
            current_weight+=x.weight
            weight.append(current_weight)
        for idx in range(len(self.memory)):
            weight[idx] = weight[idx]/current_weight
        return random.choices(
            population = list(range(len(self.memory))),
            weights = weight,
            k = batch_size 
        )
    def sample(self, batch_size):
        if len(self.memory)>batch_size:
            import random
            normal_batch = batch_size//2;
            idx_list1 = []
            for x in range(normal_batch):
                idx_list1.append(random.randint(0,normal_batch-1))
            idx_list2 = self.weight_sample(batch_size=batch_size-normal_batch)
            idx_list = idx_list1 + idx_list2
            res = []
            for idx in idx_list:
                res.append(self.memory[idx])
            return res,idx_list
        else:
            return self.memory,list(range(len(self.memory)))
    def updateWeight(self,idx_list,weight_list):
        for idx,wei in zip(idx_list,weight_list):
            # print(self.memory[idx].weight,weight_list[idx])
            self.memory[idx] = self.memory[idx]._replace(weight=wei)
            # self.memory[idx].weight = weight_list[idx]
    def __len__(self):
        return len(self.memory)
    def resetMemory(self,):
        self.memory =[]
        self.position = 0

class TreeNet:
    def __init__(self,tree_builder,value_network):
        self.tree_builder  = tree_builder#sql2fea.TreeBuilder
        self.value_network = value_network#TreeLSTM.SPINN
        self.optimizer = optim.Adam(value_network.parameters(),lr = 3e-4   ,betas=(0.9,0.999))
        self.memory = ReplayMemory(config.mem_size)
        self.loss_function = MSEVAR(config.var_weight)
        # self.loss_function = F.smooth_l1_loss
    def plan_to_value(self,tree_feature,sql_feature):
        def recursive(tree_feature):
            if isinstance(tree_feature[1],tuple):
                feature = tree_feature[0]
                h_left,c_left = recursive(tree_feature=tree_feature[1])
                h_right,c_right = recursive(tree_feature=tree_feature[2])
                return self.value_network.tree_node(h_left,c_left,h_right,c_right,feature)
            else:
                feature = tree_feature[0]
                h_left,c_left = self.value_network.leaf(tree_feature[1])
                h_right,c_right = self.value_network.zero_hc()
                return self.value_network.tree_node(h_left,c_left,h_right,c_right,feature)
        plan_feature = recursive(tree_feature=tree_feature)
        multi_value = self.value_network.logits(plan_feature[0],sql_feature)
        return multi_value
    def plan_to_value_fold(self,tree_feature,sql_feature,fold):
        def recursive(tree_feature):
            if isinstance(tree_feature[1],tuple):
                feature = tree_feature[0]
                h_left,c_left = recursive(tree_feature=tree_feature[1]).split(2)
                h_right,c_right = recursive(tree_feature=tree_feature[2]).split(2)
                return fold.add('tree_node',h_left,c_left,h_right,c_right,feature)
            else:
                feature = tree_feature[0]
                h_left,c_left = fold.add('leaf',tree_feature[1]).split(2)
                h_right,c_right= fold.add('zero_hc',1).split(2)
                return fold.add('tree_node',h_left,c_left,h_right,c_right,feature)
        plan_feature,c = recursive(tree_feature=tree_feature).split(2)
        # sql_feature = fold.add('sql_feature',sql_vec)
        multi_value = fold.add('logits',plan_feature,sql_feature)
        return multi_value
    def plan_to_value_linear_fold(self,tree_feature,sql_feature,fold):
        plan_vec = np.zeros((1,config.max_alias_num))
        def recursive(tree_feature,depth=1):
            if isinstance(tree_feature[1],tuple):
                feature = tree_feature[0]
                recursive(tree_feature=tree_feature[1],depth=depth+1)
                recursive(tree_feature=tree_feature[2],depth=depth+1)
                return
                # return fold.add('tree_node',h_left,c_left,h_right,c_right,feature)
            else:
                plan_vec[0][tree_feature[1].item()] = depth
                return
                # return fold.add('tree_node',h_left,c_left,h_right,c_right,feature)
        recursive(tree_feature=tree_feature,depth=1)
        plan_feature = torch.tensor(plan_vec,device = config.device,dtype = torch.float32).reshape(-1,config.max_alias_num)
        # sql_feature = fold.add('sql_feature',sql_vec)
        multi_value = fold.add('logits_linear',plan_feature,sql_feature)
        return multi_value
    def plan_to_value_mlp_fold(self,tree_feature,sql_feature,fold):
        plan_vec = np.zeros((1,config.max_alias_num))
        def recursive(tree_feature,depth=1):
            if isinstance(tree_feature[1],tuple):
                feature = tree_feature[0]
                recursive(tree_feature=tree_feature[1],depth=depth+1)
                recursive(tree_feature=tree_feature[2],depth=depth+1)
                return
                # return fold.add('tree_node',h_left,c_left,h_right,c_right,feature)
            else:
                plan_vec[0][tree_feature[1].item()] = depth
                return
                # return fold.add('tree_node',h_left,c_left,h_right,c_right,feature)
        recursive(tree_feature=tree_feature,depth=1)
        plan_feature = torch.tensor(plan_vec,device = config.device,dtype = torch.float32).reshape(-1,config.max_alias_num)
        # sql_feature = fold.add('sql_feature',sql_vec)
        multi_value = fold.add('logits_mlp',plan_feature,sql_feature)
        return multi_value
    def loss(self,multi_value,target,var,optimize = True):
        loss_value = self.loss_function(multi_value=multi_value, target=target,var=var)
        if not optimize:
            return loss_value.item()
        self.optimizer.zero_grad()
        loss_value.backward()
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-2, 2)
        self.optimizer.step()
        return loss_value.item()
    def mean_and_variance(self,multi_value):
        mean_value = torch.mean(multi_value,dim = 1).reshape(-1,1)
        variance = torch.sum((multi_value-mean_value)**2,dim = 1)/multi_value.shape[1]
        if mean_value.shape[0]==1:
            return mean_value.item(),variance.item()**0.5
        else:
            return mean_value.data,variance.data**0.5
    def target_feature(self,target_value):
        return self.value_network.target_vec(target_value).reshape(1,-1)
    def add_sample(self,tree_feature,sql_vec,target_value,mask,weight):
        self.memory.push(tree_feature,sql_vec,target_value,mask,weight)
    def train(self,plan_json,sql_vec,target_value,mask,is_train=False):
        tree_feature = self.tree_builder.plan_to_feature_tree(plan_json)
        # print("-----")
        # print(tree_feature[0],target_value)
        # print("-----")
        target_feature = self.target_feature(target_value)
        # print(sql_vec)
        sql_feature = self.value_network.sql_feature(sql_vec)
        multi_value = self.plan_to_value(tree_feature=tree_feature,sql_feature = sql_feature)
        loss_value = self.loss(multi_value=multi_value[:,:config.head_num]*mask,target=target_feature*mask,optimize=is_train,var = multi_value[:,config.head_num])
        mean,variance  = self.mean_and_variance(multi_value=multi_value[:,:config.head_num])
        self.add_sample(tree_feature,sql_feature,target_feature,mask,abs(mean-target_value))
        from math import e
        return loss_value,mean,variance,e**multi_value[:,config.head_num].item()
    def optimize(self):
        fold = torchfold.Fold(cuda=True)
        samples,samples_idx = self.memory.sample(config.batch_size)
        target_features = []
        masks = []
        multi_list = []
        target_values = []
        for one_sample in samples:
            # print(one_sample)
            multi_value = self.plan_to_value_fold(tree_feature=one_sample.tree_feature,sql_feature = one_sample.sql_feature,fold=fold)
            masks.append(one_sample.mask)
            target_features.append(one_sample.target_feature)
            target_values.append(one_sample.target_feature.mean().item())
            multi_list.append(multi_value)
        multi_value = fold.apply(self.value_network,[multi_list])[0]
        mask = torch.cat(masks,dim = 0)
        target_feature = torch.cat(target_features,dim=0)
        loss_value = self.loss(multi_value=multi_value[:,:config.head_num]*mask,target=target_feature*mask,optimize=True,var = multi_value[:,config.head_num])
        mean,variance  = self.mean_and_variance(multi_value=multi_value[:,:config.head_num])
        mean_list = [mean] if isinstance(mean,float) else [x.item() for x in mean]
        new_weight = [abs(x-target_values[idx])*target_values[idx] for idx,x in enumerate(mean_list)]
        self.memory.updateWeight(samples_idx,new_weight)
        return loss_value,mean,variance,torch.exp(multi_value[:,config.head_num]).data.reshape(-1)
    def optimize_mlp(self):
        fold = torchfold.Fold(cuda=True)
        samples,samples_idx = self.memory.sample(config.batch_size)
        target_features = []
        masks = []
        multi_list = []
        target_values = []
        for one_sample in samples:
            # print(one_sample)
            multi_value = self.plan_to_value_fold(tree_feature=one_sample.tree_feature,sql_feature = one_sample.sql_feature,fold=fold)
            masks.append(one_sample.mask)
            target_features.append(one_sample.target_feature)
            target_values.append(one_sample.target_feature.mean().item())
            multi_list.append(multi_value)
        multi_value = fold.apply(self.value_network,[multi_list])[0]
        mask = torch.cat(masks,dim = 0)
        target_feature = torch.cat(target_features,dim=0)
        loss_value = self.loss(multi_value=multi_value[:,:config.head_num]*mask,target=target_feature*mask,optimize=True,var = multi_value[:,config.head_num])
        mean,variance  = self.mean_and_variance(multi_value=multi_value[:,:config.head_num])
        mean_list = [mean] if isinstance(mean,float) else [x.item() for x in mean]
        new_weight = [abs(x-target_values[idx])*target_values[idx] for idx,x in enumerate(mean_list)]
        self.memory.updateWeight(samples_idx,new_weight)
        return loss_value,mean,variance,torch.exp(multi_value[:,config.head_num]).data.reshape(-1)
    def optimize_linear(self):
        fold = torchfold.Fold(cuda=True)
        samples,samples_idx = self.memory.sample(config.batch_size)
        target_features = []
        masks = []
        multi_list = []
        target_values = []
        for one_sample in samples:
            # print(one_sample)
            multi_value = self.plan_to_value_fold(tree_feature=one_sample.tree_feature,sql_feature = one_sample.sql_feature,fold=fold)
            masks.append(one_sample.mask)
            target_features.append(one_sample.target_feature)
            target_values.append(one_sample.target_feature.mean().item())
            multi_list.append(multi_value)
        multi_value = fold.apply(self.value_network,[multi_list])[0]
        mask = torch.cat(masks,dim = 0)
        target_feature = torch.cat(target_features,dim=0)
        loss_value = self.loss(multi_value=multi_value[:,:config.head_num]*mask,target=target_feature*mask,optimize=True,var = multi_value[:,config.head_num])
        mean,variance  = self.mean_and_variance(multi_value=multi_value[:,:config.head_num])
        mean_list = [mean] if isinstance(mean,float) else [x.item() for x in mean]
        new_weight = [abs(x-target_values[idx])*target_values[idx] for idx,x in enumerate(mean_list)]
        self.memory.updateWeight(samples_idx,new_weight)
        return loss_value,mean,variance,torch.exp(multi_value[:,config.head_num]).data.reshape(-1)
        
    # def predict(self,plan_json,sql_vec,target_value):
    #     tree_feature = self.tree_builder.plan_to_feature_tree(plan_json)
    #     target_feature = self.target_feature(target_value)
    #     sql_feature = self.value_network.sql_feature(sql_vec)
    #     multi_value = self.plan_to_value(tree_feature=tree_feature,sql_feature = sql_feature)
    #     loss_value = self.loss(multi_value=multi_value[:,:config.head_num],target=target_feature,optimize=False,var = multi_value[:,config.head_num])
    #     mean,variance  = self.mean_and_variance(multi_value=multi_value[:,:config.head_num])
    #     from math import e 
    #     return loss_value,mean,variance,self.value_extractor.decode(multi_value[:,config.head_num].item())
    
    # def predict(self,plan_json,sql_vec,target_value):
    #     tree_feature = self.tree_builder.plan_to_feature_tree(plan_json)
    #     target_feature = self.target_feature(target_value)
    #     sql_feature = self.value_network.sql_feature(sql_vec)
    #     multi_value = self.plan_to_value(tree_feature=tree_feature,sql_feature = sql_feature)
    #     loss_value = self.loss(multi_value=multi_value[:,:config.head_num],target=target_feature,optimize=False,var = multi_value[:,config.head_num])
    #     mean,variance  = self.mean_and_variance(multi_value=multi_value[:,:config.head_num])
    #     from math import e 
    #     return loss_value,mean,variance,self.value_extractor.decode(multi_value[:,config.head_num].item())
    # def optimize(self,batch_size):
        

MCTSTransition = namedtuple('MCTSTransition',
                    ('leading_feature', 'sql_feature', 'target_feature','weight'))

class MCTSReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        data =  MCTSTransition(*args)
        position = self.position
        self.memory[position] = data
        self.position = (self.position + 1) % self.capacity
    def weight_sample(self,batch_size):
        import random
        weight = []
        current_weight = 0
        for x in self.memory:
            current_weight+=x.weight
            weight.append(current_weight)
        for idx in range(len(self.memory)):
            weight[idx] = weight[idx]/current_weight
        return random.choices(
            population = list(range(len(self.memory))),
            weights = weight,
            k = batch_size 
        )
    def sample(self, batch_size):
        if len(self.memory)>batch_size:
            import random
            normal_batch = batch_size//2;
            idx_list1 = []
            for x in range(normal_batch):
                idx_list1.append(random.randint(0,normal_batch-1))
            idx_list2 = self.weight_sample(batch_size=batch_size-normal_batch)
            idx_list = idx_list1 + idx_list2
            res = []
            for idx in idx_list:
                res.append(self.memory[idx])
            return res,idx_list
        else:
            return self.memory,list(range(len(self.memory)))
    def updateWeight(self,idx_list,weight_list):
        for idx,wei in zip(idx_list,weight_list):
            # print(self.memory[idx].weight,weight_list[idx])
            self.memory[idx] = self.memory[idx]._replace(weight=wei)
            # self.memory[idx].weight = weight_list[idx]
    def __len__(self):
        return len(self.memory)
    def resetMemory(self,):
        self.memory =[]
        self.position = 0


class ValueNet(nn.Module):
    def __init__(self, in_dim,n_words=40,hidden_size = 64):
        super(ValueNet, self).__init__()
        self.dim = in_dim
        self.layer1 = nn.Sequential(nn.Linear(in_dim, hidden_size), nn.ReLU(True))
        # self.layer2 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True))
        # self.layer3 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(True))
        # self.layer4 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(True))
        # self.layer5 = nn.Sequential(nn.Linear(32, out_dim), nn.Softmax(dim = 0))
        self.output_layer = nn.Sequential(nn.Linear(hidden_size*2,hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size,hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size,1))
        self.table_embeddings = nn.Embedding(n_words, hidden_size)#2 * max_column_in_table * size)
        self.hs = hidden_size
        # self.layer5 = nn.Sequential(nn.Linear(32, out_dim), nn.ReLU(True))
        self.cnn = nn.Sequential(nn.Conv1d(in_channels = self.hs, out_channels = self.hs, kernel_size=5,padding=2),
                                 nn.ReLU(),
                                 nn.Conv1d(in_channels = self.hs, out_channels = self.hs, kernel_size=5,padding=2),
                                 nn.ReLU(),
                                 nn.Conv1d(in_channels = self.hs, out_channels = self.hs, kernel_size=5,padding=2),
                                 nn.MaxPool1d(kernel_size = config.max_hint_num))
        self.rnn = nn.LSTM(input_size=self.hs,hidden_size=self.hs,batch_first = True)
        # input = torch.randn(5, 3, 32)
    def forward(self, QE, JO):
        # x = x.reshape(-1, self.dim)
        # print(QE.shape,JO.shape)
        x = self.layer1(QE).reshape(-1,self.hs)
        # print(X.shape)
        # flush(stdou)
        # print(JO.dtype)
        JOE = self.table_embeddings(JO).reshape(-1,config.max_hint_num,self.hs)
        # _,(h,c) = self.rnn(JOE) 
        h = self.cnn(JOE.permute(0,2,1))
        ox = torch.cat((x,h.reshape(-1,self.hs)),dim=1)
        x = self.output_layer(ox)
        return x
