#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 20:15:50 2019

@author: yanyifan
"""

import numpy as np


class Map(object):
    """
    the object of map defining basic operations
    """
    def __init__(self):
        self.graph = [[1,1,1,1,1,1,1,1],
                      [1,0,0,0,0,0,0,1],
                      [1,0,0,0,1,0,0,1],
                      [1,0,0,0,1,0,0,1],
                      [1,0,0,0,1,0,0,1],
                      [1,0,0,0,0,0,0,1],
                      [1,1,1,1,1,1,1,1]]
        self.start = (3,2)
        self.end = (3,6)
        
        self.dir = ((-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0))
        
        self.cur = self.start
        
        #self.weight = [ [ 0 for i in range(len(self.graph[0]))] for j in range(len(self.graph)) ]
        
        self.weight = np.zeros(shape = (7,8))
    def could(self,pos:tuple):
        if pos[0]>=1 and pos[0]<len(self.graph) and pos[1]>=1 and pos[1] < len(self.graph[0]):
            return self.graph[pos[0]][pos[1]] == 0
        return False
    
    def possible_dir(self):
        ans = []
        for i in range(len(self.dir)):
            next_p = (self.cur[0]+self.dir[i][0],self.cur[1]+self.dir[i][1])
            if self.could(next_p):
                ans.append(next_p)
        return ans
    
    def random_walk(self):
        possible = self.possible_dir()
        ch = np.random.choice(len(possible))
        self.cur = possible[ch]
        return self.cur
        
    def back_reward(self,road:list):
        l = len(road)
        for i in range(l):
            pos = road[i]
            self.weight[pos[0]][pos[1]] += 1/l**2
            
        

def MC_search(itr:int,max_run:int= 20):
    """
    用MC方法搜索最短路径
    Parameters
    ----------
    itr : int
        迭代次数
    max_run : int
        最长路径限制
    Returns
    -------
        最短路路径
    """
    
    # short = None
    m = Map()
    for cnt in range(itr):
        m.cur = m.start
        road = [m.cur]
        cur = 0
        while(m.cur != m.end and cur<max_run):
            m.random_walk()
            road.append(m.cur)
            cur+=1
        if cur<max_run:
            m.back_reward(road)
        # if short == None:
        #     short = road
        # else:
        #     if len(road) < len(short):
        #         short = road
    
    final = [m.start]
    m.cur = m.start
    
    while m.cur != m.end:
        pd = m.possible_dir()
        
        max_ = -1
        next_ = None
        for item in pd:
            if m.weight[item[0]][item[1]] > max_ and (item not in final):
                max_ = m.weight[item[0]][item[1]]
                next_ = item
        m.cur = next_
        
        final.append(m.cur)
    
    return final


if __name__ == '__main__':
    q = MC_search(300)
    print(q)
    print(len(q))
    
        
        