
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@filename        :KNN.py
@readme        :
@time        :2021/11/25 21:16:32
@author        :yxfish13
@版本        :1.0
# Copyright 2018-2021 Xiang Yu(x-yu17(at)mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the 
): you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 
 BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
'''

import numpy as np
import random
class KNN:
    def __init__(self,k_neighbours,method='brute',max_distance=0.05**2+0.05**2+0.05**2+10,max_v=4):
        """[summary]

        Args:
            k_neighbours ([int]): [number of neighbours]
            method (str, optional): [description]. Defaults to 'brute'.
        """
        self.k_neighbours = k_neighbours
        self.max_distance = max_distance
        self.max_v = max_v
        self.method = method
        if method =='brute':
            self.kvs = []
        if method =='kd-tree':
            pass
    def distance(self,v1,v2):
        # print(zip(v1,v2))
        return sum([abs(v[0]-v[1]) for v in zip(v1[1:2],v2[1:2])])
    def insertValues(self,data):
        """[summary]

        Args:
            data ([list]): [description]
        """
        if self.method=='brute':
            self.kvs.extend(data)
    def insertAValue(self,data):
        """[summary]

        Args:
            data ([list]): [description]
        """
        data = (data[0],abs(data[1]))
        self.kvs.append(data)
    def kNeighbours(self,v,k_neighbours=0):
        if k_neighbours==0:
            k_neighbours = self.k_neighbours
        chosen_data = sorted([(self.distance(v,x[0]),x[1]) for x in self.kvs],key = lambda x:x[0])[:k_neighbours]
        if len(self.kvs)<k_neighbours:
            return []
        return chosen_data
    def kNeightboursSample(self,v,k_neighbours=0):
        k_neighbours = self.kNeighbours(v,k_neighbours)
        if len(k_neighbours)==0 or k_neighbours[-1][0]>self.max_distance:
            return self.max_v
        return random.sample(k_neighbours,1)[0][1]
    
    


if __name__ == "__main__":
    origin_data = [((1,2,3),1),
                   ((2,3,4),2),
                   ((3,4,5),3),
                   ((4,5,6),4)]
    target_data = (2,2,2)
    knn = KNN(2)
    knn.insertValues(origin_data)
    print(knn.kNeighbours(target_data))
    knn.insertAValue(((2,2,3),5))
    print(knn.kNeighbours(target_data))