from os import DirEntry
import torch
from torch.nn import init
import torchfold
import torch.nn as nn
from ImportantConfig import Config

config = Config()

class MSEVAR(nn.Module):
    def __init__(self,var_weight):
        super(MSEVAR, self).__init__()
        self.var_weight = var_weight

    def forward(self, multi_value, target,var):
        var_wei = (self.var_weight * var).reshape(-1,1)
        loss1 = torch.mul(torch.exp(-var_wei), (multi_value - target) ** 2)
        loss2 = var_wei
        loss3 = 0
        loss = (loss1 + loss2 + loss3)
        return loss.mean()
    
class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.fc_left = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_right = nn.Linear(hidden_size, 5 * hidden_size)
        self.fc_input = nn.Linear(input_size, 5 * hidden_size)
        elementwise_affine = False
        self.layer_norm_input = nn.LayerNorm(5 *hidden_size,elementwise_affine = elementwise_affine)
        self.layer_norm_left = nn.LayerNorm(5 *hidden_size,elementwise_affine = elementwise_affine)
        self.layer_norm_right = nn.LayerNorm(5 *hidden_size,elementwise_affine = elementwise_affine)
        self.layer_norm_c = nn.LayerNorm(hidden_size,elementwise_affine = elementwise_affine)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, h_left,c_left,h_right,c_right,feature):
        lstm_in = self.layer_norm_left(self.fc_left(h_left))
        lstm_in += self.layer_norm_right(self.fc_right(h_right))
        lstm_in += self.layer_norm_input(self.fc_input(feature))
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * c_left +
             f2.sigmoid() * c_right)
        c = self.layer_norm_c(c)
        h = o.sigmoid() * c.tanh()
        return h,c
    def zero_h_c(self,input_dim = 1):
        return torch.zeros(input_dim,self.hidden_size,device = config.device),torch.zeros(input_dim,self.hidden_size,device = config.device)
class Head(nn.Module):
    def __init__(self,hidden_size):
        super(Head, self).__init__()
        self.hidden_size = hidden_size
        self.head_layer = nn.Sequential( nn.Linear(hidden_size*2,hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size,1),
                                         )
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.head_layer(x)
        return out
        
class SPINN(nn.Module):

    def __init__(self, head_num, input_size, hidden_size, table_num,sql_size):
        super(SPINN, self).__init__()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.table_num = table_num
        self.input_size = input_size
        self.sql_size = sql_size
        self.tree_lstm = TreeLSTM(input_size = input_size,hidden_size = hidden_size)
        self.sql_layer = nn.Linear(sql_size,hidden_size)
        
        self.head_layer = nn.Sequential( nn.Linear(hidden_size*2,hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size,head_num+1),
                                        )
        self.table_embeddings = nn.Embedding(table_num, hidden_size)#2 * max_column_in_table * size)
        self.heads = nn.ModuleList([Head(self.hidden_size) for i in range(self.head_num+1)])
        self.relu = nn.ReLU()

    def leaf(self, alias_id):
        table_embedding  = self.table_embeddings(alias_id)
        return (table_embedding, torch.zeros(table_embedding.shape,device = config.device,dtype = torch.float32))
    def input_feature(self,feature):
        return torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(-1,self.input_size)
    def sql_feature(self,feature):
        return torch.tensor(feature,device = config.device,dtype = torch.float32).reshape(1,-1)
    def target_vec(self,target):
        return torch.tensor([target]*self.head_num,device = config.device,dtype = torch.float32).reshape(1,-1)
    def tree_node(self, h_left,c_left,h_right,c_right,feature):
        h,c =  self.tree_lstm(h_left,c_left,h_right,c_right,feature)
        return (h,c)
    
    def logits(self, encoding,sql_feature,prt=False):
        sql_hidden = self.relu(self.sql_layer(sql_feature))
        out_encoding = torch.cat([encoding,sql_hidden],dim = 1)
        out = self.head_layer(out_encoding)
        return out
    
    def zero_hc(self,input_dim = 1):
        return (torch.zeros(input_dim,self.hidden_size,device = config.device),torch.zeros(input_dim,self.hidden_size,device = config.device))
    
        
