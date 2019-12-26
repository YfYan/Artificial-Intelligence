#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:28:51 2019

@author: yanyifan
"""

from MC_search import Map
import copy
import sys
import numpy as np

np.random.seed(0)
class MCTS_node(object):
    
    def __init__(self,parent,pos,road):
        self.parent:MCTS_node = parent
        self.pos = pos
        self.child = []
        self.n = 1
        self.r = 0
        self.road = copy.deepcopy(road)
        self.expanded = False
        self.m = Map()
        self.m.cur = pos
        for item in road:
            self.m.graph[item[0]][item[1]] = 1
    
    
    def expansion(self):
        """
        节点扩展
        """
        possible = self.m.possible_dir()
        for pos in possible:
            next_road = copy.deepcopy(self.road)
            next_road.append(pos)
            self.child.append(MCTS_node(self,pos,next_road))
        self.expanded = True
    
    def selection(self,total):
        """
        选择子节点
        """
        max_ = - sys.maxsize
        ans = None
        l = []
        for ch in self.child:
            cur = ch.r/ch.n + 10*np.sqrt(total/ch.n)
            if cur > max_+1e-6:
                max_ = cur
                #ans = ch
                l = [ch]
            elif np.abs(cur - max_) <1e-6:
                l.append(ch)
        r = np.random.randint(low = 0,high = len(l))
        return l[r]
     
    def back_ward(self,res):
        self.n+=1
        self.r+=res
        if self.parent != None:
            self.parent.back_ward(res)
            

def mcts_simulation(itr:int):
    """

    Parameters
    ----------
    itr : int
        指定要搜索的次数
    Returns
    -------
    road : list
        目前找到的最短路
    """
    road = None
    root:MCTS_node = MCTS_node(None,(3,2),[(3,2)])
    
    for i in range(itr):
        p:MCTS_node = root
        flag = False
        while p.expanded == False or len(p.child) != 0:
            if p.pos == (3,6):
                p.back_ward(10/len(p.road)**2)
                flag = True
                if road == None:
                    road = p.road
                else:
                    if len(p.road) < len(road):
                        road = p.road
                break
            if p.expanded == False:
                p.expansion()
            if len(p.child)!=0:
                p = p.selection(i+1)
            
        if flag == False:
            p.back_ward(0)       
    return road
        

if __name__ == '__main__':
    road = mcts_simulation(200)
    print(road)
    print(len(road))