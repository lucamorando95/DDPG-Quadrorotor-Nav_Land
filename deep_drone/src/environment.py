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
from geometry_msgs.msg import Twist,PoseWithCovariance, Quaternion, Point, Pose, Vector3, Vector3Stamped, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import String, Header
from std_msgs.msg import Empty 
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
#from keras.layers.core import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import math
import pdb


#  write here all the parameters
class Environment():
    def __init__(self, debug, goalPosition):
        
        self.gazebo = gazebo.GazeboInterface()
#      connction to  velocity topic
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
#       Define max and min velocity command 
        self.max_vel = 1.5
        self.min_vel = -1.5

#      Define Goal Position in world frame 
        self.goalPosition = goalPosition#[2.0, 3.0] #the position is defined only on x,y coordinates 
        
        self.goal_threshold = 0.7 #0.9
        self.reward_crash = -5
        self.goal_reward = 50
        self.goal_reached = 0
        
        self.num_states = 2 #(x_des - x) and (y_des - y)
        self.num_actions = 2
        
        #Define max tollerance for position in order to end episode 
        self.max_x = 10
        self.min_x = -10
        
        self.max_y = 10
        self.min_y = -10
        self.max_incl = np.pi/3
        #take into account previous state and previous reward 
        self.prev_state = []
        self.prev_reward = []
        
        self.linear_vel_z = 0.0
        self.angular_vel_z  = 0.0
       
        self.step_running = 0.05 #required for rod param conversion
        
        self.debug = debug
        
        
    def _reset(self, test_flag):
        #reset Simulation to Initial Value
         #resetSim = False
         TakeOff = False
         self.gazebo.resetSim()
         
         
        #unpause simulation: if paused
         self.gazebo.unpauseSim()
         
        #Repeat TakeOff procedure
         
         uav = AutonomousFlight()
         while not rospy.is_shutdown():
            if TakeOff == False:
               takeOff = uav.SendTakeOff(TakeOff, uav, test_flag)
               TakeOff = takeOff
              
            else:
               break
         
        #Get Initial states 
         initPoseData, imuData, velData,_ = self.takeEnvObservations(test_flag)
        
         initPoseStateData = [initPoseData.pose.pose.position.x, initPoseData.pose.pose.position.y] #take the position of the drone respect ground
         self.plotState = np.asarray(initPoseStateData)
         self.prev_state = initPoseStateData
        
        #Pauses Simulation 
         self.gazebo.pauseSim()
            
         return initPoseStateData
        
    def _step(self, action,step, test_flag): #NB dovranno essere aggiunte anche le informazioni relative al veicolo terrestre 
        
        #Take as Input Action
        #Give as Output nextState, reward, Terminal 
        
        vel = Twist()
        vel.linear.x = action[0]
        vel.linear.y = action[1]
        vel.linear.z = self.linear_vel_z
        vel.angular.z = self.angular_vel_z 
        if self.debug:
            print('vel_x:{}, vel_y: {}'.format(vel.linear.x, vel.linear.y))
            #if there are other things to be printed
        
        ## Transformation of velocity from body frame to world frame in the case of testing
        if test_flag:
           poseData,imuData, velData, altitudeVelDrone = self.takeEnvObservations(test_flag)
           _, _, _, R_tot = self.from_quaternion_to_euler(poseData, imuData, velData, step, test_flag)
           vel_vect = [vel.linear.x, vel.linear.y, vel.linear.z]
           vel_vect = np.transpose(vel_vect)
           vel_world = [[0],
                        [0],
                        [0]]
           R_tot = np.array(R_tot)
          
          
           for i in range(R_tot.shape[0]): #row
               for j in range(R_tot.shape[1]): #column
                vel_world[i] +=  R_tot[i][j] * vel_vect[j] 
           vel.linear.x  =  vel_world[0]
           vel.linear.y = vel_world[1]
           print('vel.linear.x', vel.linear.x)
           print(' vel.linear.y ',  vel.linear.y )
        
        
        self.gazebo.unpauseSim()
        
        self.pub.publish(vel)
        time.sleep(self.step_running) #stop the code in order to wait the upgrade of the simulation phisycs
        poseData,imuData, velData, altitudeVelDrone = self.takeEnvObservations(test_flag) #it is possible to add imu inf, ecc
        self.gazebo.pauseSim()
        if test_flag:
           self.linear_vel_z = altitudeVelDrone.linear.z
           self.angular_vel_z = altitudeVelDrone.angular.z
        #Do the processing of the data and the state reached in oreder to obtain a reward  and define if it is the terminal state or not
        reward, isTerminal= self.processingData(poseData, imuData, velData, step, test_flag)
        
        nextState = [poseData.pose.pose.position.x, poseData.pose.pose.position.y]
        self.prev_state = nextState
        #self.plotState = np.vstack((self.plotState, np.asarray(nextState))) #cretate a vertical array of two array concatenated 
        altitude = poseData.pose.pose.position.z
        if isTerminal[0] == True:
            self.goal_reached = self.goal_reached + 1
            print('State terminal reached number :', self.goal_reached)
           
        return nextState, reward, isTerminal, altitude
     
    def from_quaternion_to_euler(self,poseData, imuData, velData, step, test_flag):
        a1 = +2.0 * (imuData.orientation.w * imuData.orientation.x + imuData.orientation.y * imuData.orientation.z)
        a2 = 1.0 - 2.0 *(imuData.orientation.x*imuData.orientation.x + imuData.orientation.y*imuData.orientation.y)
        roll = math.atan2(a1, a2) #theta
        
        #pitch angle --> y axis rotation
        a3 = 2.0*(imuData.orientation.w* imuData.orientation.y - imuData.orientation.z * imuData.orientation.x)
        if a3 > 1:
            a3 = 1
        if a3 < -1:
            a3 = -1
        pitch = math.asin(a3) #phi
        
        #yaw angle (z axis rotation)
        a4 = 2.0 * (imuData.orientation.w * imuData.orientation.z + imuData.orientation.x * imuData.orientation.y)        
        a5 = 1.0 - 2.0* (imuData.orientation.y * imuData.orientation.y + imuData.orientation.z * imuData.orientation.z)
        yaw = math.atan2(a4, a5) #psi
        
        #Rotating velocity to world frame if YAw is aligned to vehicle 
        if test_flag:
#            R_theta = [[1, 0, 0],
#                       [0, math.cos(roll), -math.sin(roll)],
#                       [0, math.sin(roll), math.cos(roll)]]
#            R_phi = [[math.cos(pitch), 0, math.sin(pitch)],
#                      [0, 1, 0],
#                      [-math.sin(pitch), 0, math.cos(pitch)]]
#            R_yaw =  [[math.cos(yaw), -math.sin(yaw), 0],
#                      [math.sin(yaw), math.cos(yaw), 0],
#                      [0, 0, 1]]
#            R_tot = R_theta * R_phi * R_yaw
            R_tot = [[math.cos(pitch)*math.cos(yaw), math.cos(pitch)*math.sin(yaw),  -math.sin(pitch)],
                      [math.sin(roll)*math.sin(pitch)*math.cos(yaw) - math.cos(roll)*math.sin(yaw) , math.sin(roll)*math.sin(pitch)*math.sin(yaw) + math.cos(roll)*math.cos(yaw), math.sin(roll)*math.cos(pitch)],
                      [math.cos(roll)*math.sin(pitch)*math.cos(yaw) + math.sin(pitch)*math.sin(yaw), math.cos(roll)* math.sin(pitch)*math.sin(yaw) -  math.sin(roll)*math.cos(yaw),  math.cos(roll)*math.cos(pitch)]]
            
        return roll, pitch, yaw, R_tot
    
    def takeEnvObservations(self, test_flag): #Function which takes information from the environment in gazebo 
        self.gazebo.unpauseSim()
        #Take pose information 
        poseData = None
        while poseData is None :
            try:
                poseData = rospy.wait_for_message('/ground_truth/state', Odometry, timeout = 5)
            except:
                rospy.loginfo('Unable to reach the drone Pose topic. Try to connect again')
                
        
        imuData = None 
        while imuData is None :
            try:
                imuData = rospy.wait_for_message('/ardrone/imu', Imu, timeout = 5)
            except:
                rospy.loginfo('Unable to reach the drone Imu topic. Try to connect again')
        
        velData = None
        while velData is None:
          try:
              velData = rospy.wait_for_message('/fix_velocity', Vector3Stamped, timeout=5)
          except:
              rospy.loginfo("Unable to reach the drone Imu topic. Try to connect again")
        altitudeVelDrone = None
        if test_flag == True:
          
           while altitudeVelDrone is None:
             try:
                 altitudeVelDrone = rospy.wait_for_message('/drone/cmd_vel', Twist, timeout=5)
                
             except:
                 rospy.loginfo("Unable to reach the drone velocity topic. Try to connect again")      
              
    
        return poseData, imuData, velData, altitudeVelDrone
    
   
    
    def processingData(self,poseData, imuData, velData, step, test_flag):
        #This Function takes the information obtained from the environment and check if they are inside the limit defined for a good learning.
        Terminal = [False]
        roll, pitch, yaw, R_tot = self.from_quaternion_to_euler(poseData, imuData, velData, step, test_flag)
#        R_tot = []
#        #Obtain eulerian angles from quaternion infromation given by imuData
#        #roll angle --> x axis rotation
#        
#        a1 = +2.0 * (imuData.orientation.w * imuData.orientation.x + imuData.orientation.y * imuData.orientation.z)
#        a2 = 1.0 - 2.0 *(imuData.orientation.x*imuData.orientation.x + imuData.orientation.y*imuData.orientation.y)
#        roll = math.atan2(a1, a2) #theta
#        
#        #pitch angle --> y axis rotation
#        a3 = 2.0*(imuData.orientation.w* imuData.orientation.y - imuData.orientation.z * imuData.orientation.x)
#        if a3 > 1:
#            a3 = 1
#        if a3 < -1:
#            a3 = -1
#        pitch = math.asin(a3) #phi
#        
#        #yaw angle (z axis rotation)
#        a4 = 2.0 * (imuData.orientation.w * imuData.orientation.z + imuData.orientation.x * imuData.orientation.y)        
#        a5 = 1.0 - 2.0* (imuData.orientation.y * imuData.orientation.y + imuData.orientation.z * imuData.orientation.z)
#        yaw = math.atan2(a4, a5) #psi
#        
#        #Rotating velocity to world frame if YAw is aligned to vehicle 
#        if test_flag:
#            R_theta = [[1, 0, 0],
#                       [0, math.cos(roll), -math.sin(roll)],
#                       [0, math.sin(roll), math.cos(roll)]]
#            R_phi = [[math.cos(pitch), 0, math.sin(pitch)],
#                      [0, 1, 0],
#                      [-math.sin(pitch), 0, math.cos(pitch)]]
#            R_yaw =  [[math.cos(yaw), -math.sin(yaw), 0],
#                      [math.sin(yaw), math.cos(yaw), 0],
#                      [0, 0, 1]]
#            R_tot = R_theta * R_phi * R_yaw
            
        if pitch > self.max_incl or pitch < -self.max_incl:
            if self.debug:
                rospy.loginfo("Terminating Episode: Pitch value out of limits, unstable quad ----> "+str(pitch))
            
            if self.prev_reward is Empty:
                 self.prev_reward = 0
                 
            
            
            Terminal = [True]
            reward = self.reward_crash + self.prev_reward
            reward = [reward]
        
        elif (pitch > self.max_incl) or (pitch < -self.max_incl):
            if self.debug:
                rospy.loginfo("Terminating Episode: Roll value out of limits, unstable quad ----> "+str(roll))
            
            if self.prev_reward is Empty:
                 self.prev_reward = 0
             
            Terminal = [True]
            reward = self.reward_crash + self.prev_reward
            reward = [reward]
            
        elif poseData.pose.pose.position.x > self.max_x or poseData.pose.pose.position.x < -self.max_x:
            if self.debug:
                rospy.loginfo("Terminating Episode: X position value out of limits, unstable quad ----> "+str(poseData.pose.pose.position.x))
             
            if self.prev_reward is Empty:
                 self.prev_reward = 0
              
            Terminal = [True]
            reward = self.reward_crash + self.prev_reward
            reward = [reward]
            
        elif poseData.pose.pose.position.y > self.max_y or poseData.pose.pose.position.y < -self.max_y:
            if self.debug:
                rospy.loginfo("Terminating Episode: Y position value out of limits, unstable quad ----> "+str(poseData.pose.pose.position.y))
            
            if self.prev_reward is Empty:
                 self.prev_reward = 0
             
            Terminal = [True]
            reward = self.reward_crash + self.prev_reward
            reward = [reward]
             
        else:
            reward, goal_reached = self.getting_reward(poseData, imuData, velData,step) #To be designed the reward function
            if goal_reached:
                print('Goal Reached!')
                Terminal = [True]
        
        #Anyway it goed the reward is printed 
#        if self.debug:
#            print('Reward: {}'.format(reward))
#            print('Terminal',Terminal)
        return reward, Terminal
    
    
             
    def getting_reward(self, poseData, imuData, velData, step):
        #Takes as input the state of the drone, takenm from the sensor measurements 
        #the output is the reward evaluated when the state is not final 
        if step == 1:
            self.prev_reward = 0
            reward = 0
            reward = [reward]
        
        goal = False
        distance_error = 0
        #Evaluate distance error between the drone position on x,y and the goal position on x,y 
        x_drone = poseData.pose.pose.position.x
        y_drone = poseData.pose.pose.position.y
        z_drone = poseData.pose.pose.position.z
        print('altitude : {}'.format(z_drone))
        x_goal = self.goalPosition[0]
        y_goal = self.goalPosition[1]
        
        error_x = (x_goal - x_drone)
        error_y = (y_goal - y_drone)
        distance_error = math.sqrt(error_x*error_x + error_y*error_y)
        
#        if self.debug:
#            print('Distance Error: {}'.format(distance_error))
        
        if distance_error < self.goal_threshold:
            reward = self.goal_reward -self.prev_reward 
            reward = [reward]
            reward_t = reward
            goal = True
       
        else: #evaluate reward with the desired function
            reward    = -100*distance_error
            reward_t = reward
            reward = reward -  self.prev_reward
            reward = [reward]
        #print('Goal',goal)
        
            
        self.prev_reward = reward_t
        return reward, goal
        
        
        
  
        
        
