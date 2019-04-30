#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:00:58 2019

@author: parallels
"""

import random
import numpy as np
from collections import deque #Provides container options with fast appends and pop methods

class ReplayBuffer(object):
    def __init__(self, buffer_size = 3000, mini_batch_size = 32):
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        max_len = self.buffer_size
        self.buff = deque([], max_len) #create an empty queue 
        
    def add_experience(self,state, action, reward, state_t1, terminal):
        experience = (state, action, reward, state_t1, terminal)
        self.buff.append(experience) #add element to the right side of the queue
        
    def take_experience(self):
        #inizialize empty vector fro the sample taken from batch
        mini_batch = []
        if len(self.buff) < self.buffer_size:
            mini_batch = random.sample(list(self.buff), len(self.buff)) #list create a data structure able to store multiple data at the same time
        else:
            mini_batch = random.sample(list(self.buff), self.mini_batch_size)
        
        return mini_batch
    
    def clear_buff(self):
        self.buff.clear()