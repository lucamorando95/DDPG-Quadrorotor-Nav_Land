#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:46:54 2019

@author: parallels
"""
import rospy 
import time

import gazeboInterface as gazebo

from AutonomousFlight import AutonomousFlight
from geometry_msgs.msg import Twist,PoseWithCovariance
from std_msgs.msg import String 
from std_msgs.msg import Empty 
from nav_msgs.msg import Odometry
from keras.layers.core import Dense, Dropout, Activation, Flatten

#  write here all the parameters
class Environment():
    def __init__(self):
        
        self.gazebo = gazebo.GazeboInterface()
#      connction to  velocity topic
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
#       Define max and min velocity command 
        self.max_vel = 1.5
        self.min_vel = -1.5

#      Define Goal Position in world frame 
        self.goalPosition = [2.0, 3.0] #the position is defined only on x,y coordinates 
        
        self.reward_goal_threshold = 1
        self.reward_crash = -5
        self.goal_reward = 5
        
        self.num_states = 2 #(x_des - x) and (y_des - y)
        self.num_actions = 2
        
        #Define max tollerance for position in order to end episode 
        self.max_y = 10
        self.min_x = -10
        
        self.max_y = 10
        self.min_y = -10
        
        #take into account previous state and previous reward 
        self.prev_state = []
        self.prev_reward = []
        
        self.step_running = 0.05 #required for rod param conversion
        
    def _reset(self):
        #reset Simulation to Initial Value
         resetSim = False
         TakeOff = False
         resetSim = self.gazebo.resetSim()
         print("resetSim",resetSim)
        #unpause simulation: if paused
        #self.gazebo.unpauseSima()
         
        #Repeat TakeOff procedure
         
         uav = AutonomousFlight()
         while not rospy.is_shutdown():
            if TakeOff == False:
               takeOff = uav.SendTakeOff(TakeOff, uav)
               TakeOff = takeOff
               print("TakeOff",takeOff)
            else:
               break
         
            
         return resetSim
        

  
        
        
