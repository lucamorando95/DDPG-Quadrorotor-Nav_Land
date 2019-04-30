#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:28:30 2019

@author: parallels
"""
import rospy
from std_srvs.srv import Empty

#This script provides a connection to gazebo in order to abilitate the command pause, unpause and reset the simulation

class GazeboInterface():
    
    def __init__(self):
        
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.reset = rospy.ServiceProxy('gazebo/reset_world', Empty)
        
    def resetSim(self):
        #resetSimulation = False
        rospy.wait_for_service('/gazebo/reset_world')
        try:
            self.reset()
            #resetSimulation = True 
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service failed")
        
        #return resetSimulation
    
    def pauseSim(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")
    
    def unpauseSim(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed") 
                
        
                
        
