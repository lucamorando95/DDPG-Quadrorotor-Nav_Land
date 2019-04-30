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
    def __init__(self, debug):
        
        self.gazebo = gazebo.GazeboInterface()
#      connction to  velocity topic
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
#       Define max and min velocity command 
        self.max_vel = 1.5
        self.min_vel = -1.5

#      Define Goal Position in world frame 
        
        
        self.goal_threshold = 0.7
        self.reward_crash = -5
        self.goal_reward = 70
        self.goal_reached = 0;
        
        self.num_states = 4 #(x_des - x) and (y_des - y) and (vdes- vx) (vy_des -vy)
        self.num_actions = 2
        
        #Define max tollerance for position in order to end episode 
        self.max_x = 20
        self.min_x = -20
        
        self.max_y = 20
        self.min_y = -20
        self.max_distance_vector = 7 #max distance between the aerial vehicle and the ground target 
        
        self.max_incl = np.pi/3
        #take into account previous state and previous reward 
        self.prev_state = []
        self.prev_reward = 0.0
       
        self.drone_yaw = 0.0
        self.step_running = 0.05 #required for rod param conversion
        
        self.debug = debug
        self.count = 0
        self.start  = 0
        
    def _reset(self):
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
               takeOff = uav.SendTakeOff(TakeOff, uav)
               TakeOff = takeOff
              
            else:
               break
         
        #Get Initial states 
         initPoseData, imuData, velData, poseBox, velBox, _ = self.takeEnvObservations()
         vel_drone_x = velData.vector.x
         vel_drone_y = velData.vector.y
         x_box = poseBox.pose.pose.position.x
         y_box = poseBox.pose.pose.position.y
         
         pose_error_x =  x_box - initPoseData.pose.pose.position.x
         pose_error_y =  y_box - initPoseData.pose.pose.position.y
         #pose_error_distance =  math.sqrt(pose_error_x*pose_error_x + pose_error_y*pose_error_y)
        
         
         vel_error_x = velBox.linear.x - vel_drone_x
         vel_error_y = velBox.linear.y - vel_drone_y
         #vel_error_vehicles =  math.sqrt(vel_error_x*vel_error_x + vel_error_y*vel_error_y)
         #efine the new state vetctor : pose errorx pose error_y vel_err_X vel_err_y
        
         initPoseStateData = [pose_error_x,pose_error_y,vel_error_x, vel_error_y] #take the position of the drone respect ground
         self.plotState = np.asarray(initPoseStateData)
         self.prev_state = initPoseStateData
         
         self.prev_reward = 0.0 
        #Pauses Simulation 
         self.gazebo.pauseSim()
            
         return initPoseStateData
        
    def _step(self, action,step): #NB dovranno essere aggiunte anche le informazioni relative al veicolo terrestre 
        
        #Take as Input Action
        #Give as Output nextState, reward, Terminal 
        
        vel = Twist()
        vel.linear.x = action[0]
        vel.linear.y = action[1]
        vel.angular.z =  self.drone_yaw 
        
        if self.debug:
            print('vel_x:{}, vel_y: {}, yaw_vel: {}'.format(vel.linear.x, vel.linear.y, vel.angular.z ))
            #if there are other things to be printed
        
        self.gazebo.unpauseSim()
        
        self.pub.publish(vel)
        time.sleep(self.step_running) #stop the code in order to wait the upgrade of the simulation phisycs
        poseData,imuData, velData, poseBox, velBox, altitude_yaw_vel_drone = self.takeEnvObservations() #it is possible to add imu inf, ecc
        self.gazebo.pauseSim()
        
        #Do the processing of the data and the state reached in oreder to obtain a reward  and define if it is the terminal state or not
        reward, isTerminal = self.processingData(poseData, imuData, velData, poseBox, velBox, step)
        vel_drone_x = velData.vector.x
        vel_drone_y = velData.vector.y
        self.drone_yaw = altitude_yaw_vel_drone.angular.z
        
        x_box = poseBox.pose.pose.position.x
        y_box = poseBox.pose.pose.position.y
        
        
        pose_error_x =  x_box - poseData.pose.pose.position.x
        pose_error_y =  y_box - poseData.pose.pose.position.y
       
        vel_error_x = velBox.linear.x - vel_drone_x
        vel_error_y = velBox.linear.y - vel_drone_y
         
         #efine the new state vetctor : pose errorx pose error_y vel_err_X vel_err_y
        
        nextState = [pose_error_x,pose_error_y,vel_error_x, vel_error_y] #take the position of the drone respect ground
        
        self.prev_state = nextState
        #self.plotState = np.vstack((self.plotState, np.asarray(nextState))) #cretate a vertical array of two array concatenated 
       
        if isTerminal[0] == True:
            self.goal_reached = self.goal_reached + 1
            print('State terminal reached number :', self.goal_reached)
           
        return nextState, reward, isTerminal
    
    def takeEnvObservations(self): #Function which takes information from the environment in gazebo 
        
        #Take pose information 
        poseData = None
        while poseData is None :
            try:
                poseData = rospy.wait_for_message('/ground_truth/state', Odometry, timeout = 5)
            except:
                rospy.loginfo('Inable to reach the drone Pose topic. Try to connect again')
                
        
        imuData = None 
        while imuData is None :
            try:
                imuData = rospy.wait_for_message('/ardrone/imu', Imu, timeout = 5)
            except:
                rospy.loginfo('Inable to reach the drone Imu topic. Try to connect again')
        
        velData = None
        while velData is None:
          try:
              velData = rospy.wait_for_message('/fix_velocity', Vector3Stamped, timeout=5)
          except:
              rospy.loginfo("Inable to reach the drone Imu topic. Try to connect again")
        poseBox = None      
        while poseBox is None:
             try:
                poseBox = rospy.wait_for_message('/odom', Odometry, timeout = 5)
             except:
                rospy.loginfo('Inable to reach the Box Pose topic. Try to connect again')
        velBox = None
        while velBox is None:
             try:
                velBox = rospy.wait_for_message('/mobile/cmd_vel', Twist, timeout = 5)
             except:
                rospy.loginfo('Unable to reach the Box Velocity topic. Try to connect again')
                
        altitude_yaw_vel_drone = None
        while altitude_yaw_vel_drone is None:
             try:
                altitude_yaw_vel_drone = rospy.wait_for_message('/drone/cmd_vel', Twist, timeout = 5)
             except:
                rospy.loginfo('Unable to reach the Box Velocity topic. Try to connect again')
        
        return poseData, imuData, velData, poseBox, velBox, altitude_yaw_vel_drone
    
    def processingData(self,poseData, imuData, velData, poseBox, velBox, step):
        #This Function takes the information obtained from the environment and check if they are inside the limit defined for a good learning.
        Terminal = [False]
        
        #Obtain eulerian angles from quaternion infromation given by imuData
        #roll angle --> x axis rotation
        
        a1 = +2.0 * (imuData.orientation.w * imuData.orientation.x + imuData.orientation.y * imuData.orientation.z)
        a2 = 1.0 - 2.0 *(imuData.orientation.x*imuData.orientation.x + imuData.orientation.y*imuData.orientation.y)
        roll = math.atan2(a1, a2)
        
        #pitch angle --> y axis rotation
        a3 = 2.0*(imuData.orientation.w* imuData.orientation.y - imuData.orientation.z * imuData.orientation.x)
        if a3 > 1:
            a3 = 1
        if a3 < -1:
            a3 = -1
        pitch = math.asin(a3)
        
        #yaw angle (z axis rotation)
        a4 = 2.0 * (imuData.orientation.w * imuData.orientation.z + imuData.orientation.x * imuData.orientation.y)        
        a5 = 1.0 - 2.0* (imuData.orientation.y * imuData.orientation.y + imuData.orientation.z * imuData.orientation.z)
        yaw = math.atan2(a4, a5)
        
        #Compute distance vector between aerial vehicle and ground target
        error_x =  poseBox.pose.pose.position.x - poseData.pose.pose.position.x
        error_y =  poseBox.pose.pose.position.y - poseData.pose.pose.position.y
        distance =  math.sqrt(error_x*error_x + error_y*error_y)
       
        
        if pitch > self.max_incl or pitch < -self.max_incl:
            if self.debug:
                rospy.loginfo("Terminating Episode: Pitch value out of limits, unstable quad ----> "+str(pitch))
            
            if self.prev_reward is Empty:
                 self.prev_reward = 0 
                 
            
            
            Terminal = [True]
            reward = self.reward_crash + self.prev_reward[0]
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
        elif   distance > self.max_distance_vector:
            if self.debug:
                rospy.loginfo("Terminating Episode: Distance value between the aerial and ground vehicle out of limits, unstable quad ----> "+str(distance))
            
            if self.prev_reward is Empty:
                 self.prev_reward = 0
             
            Terminal = [True]
            reward = self.reward_crash + self.prev_reward
            reward = [reward]
        else:
            reward, goal_reached = self.getting_reward(poseData, imuData, velData, poseBox, velBox, step) #To be designed the reward function
            if goal_reached:
                print('Goal Reached!')
                Terminal = [True]
        
        #Anyway it goed the reward is printed 
#        if self.debug:
#            print('Reward: {}'.format(reward))
#            print('Terminal',Terminal)
        return reward, Terminal
    
    
             
    def getting_reward(self, poseData, imuData, velData, poseBox, velBox, step):
        #Takes as input the state of the drone, takenm from the sensor measurements 
        #the output is the reward evaluated when the state is not final 
        if step == 1:
            self.prev_reward = 0
            reward = 0
            reward = [reward]
            reward_t  = 0
       
        goal = False
        reward = 0.0
        distance_error = 0
        #Evaluate distance error between the drone position on x,y and the goal position on x,y 
        x_drone = poseData.pose.pose.position.x
        y_drone = poseData.pose.pose.position.y
        z_drone = poseData.pose.pose.position.z
        
        vel_drone_x = velData.vector.x
        vel_drone_y = velData.vector.y
        x_box = poseBox.pose.pose.position.x 
        y_box = poseBox.pose.pose.position.y 
        vel_box_x = velBox.linear.x
        vel_box_y = velBox.linear.y
        
      
#        print('x_box', x_box)
#        print('y_box', y_box)
#        print('x_drone', x_drone)
#        print('y_drone', y_drone)
        pose_error_x =  x_box - x_drone
        
        
        pose_error_y =  y_box - y_drone
       
        distance_error =  math.sqrt(pose_error_x*pose_error_x + pose_error_y*pose_error_y)
        
         
        vel_error_x = vel_box_x - vel_drone_x
        vel_error_y = vel_box_y - vel_drone_y
        vel_error_vehicles =  math.sqrt(vel_error_x*vel_error_x + vel_error_y*vel_error_y)
        print('altitude : {}'.format(z_drone))
       
        
        
#        if self.debug:
#            print('Distance Error: {}'.format(distance_error))
        print('distance_error', distance_error)
        if distance_error < self.goal_threshold:
            if self.count == 0:
                if self.prev_reward is Empty:
                    self.prev_reward = reward_t
                reward =  self.prev_reward + 5
                reward_t = reward
                reward = [reward]
                self.count = step + 1
                self.start = time.time()
               
                print('self.time',  self.start)
            elif self.count == step:
                end = time.time()
                time_passed = end - self.start
                if time_passed <= 3:
                    reward = self.prev_reward+ 5
                    reward_t = reward
                    reward = [reward]
                    self.count = step + 1
                    print('primo if self.time', time_passed)
                elif time_passed <= 5 and time_passed > 3:
                    reward = self.prev_reward +7
                    reward_t = reward
                    reward = [reward]
                    self.count = step + 1
                    print('secondo if self.time', time_passed)
                elif time_passed <= 8 and time_passed > 5:
                    reward = self.prev_reward + 10
                    reward_t = reward
                    reward = [reward]
                    self.count = step + 1
                    print('terzo if self.time', time_passed)
                elif time_passed <= 12 and time_passed > 8:
                    reward = self.prev_reward + 20
                    reward_t = reward
                    reward = [reward]
                    self.count = step + 1
                    print('quarto if self.time', time_passed)
                elif time_passed <= 17 and time_passed > 12:
                    reward = self.prev_reward + 35
                    reward_t = reward
                    reward = [reward]
                    self.count = step + 1
                    print('quinto if self.time', time_passed)
                elif time_passed > 17:
                    reward = self.prev_reward+ self.prev_reward
                    reward_t = reward
                    reward = [reward]
                    self.count = 0
                    end 
                    print('goal if self.time', time_passed)
                    goal = True
               
       
        else: #evaluate reward with the desired function
            self.count = 0
            reward    = -100*distance_error -10* vel_error_vehicles
            reward_t = reward
            print('self.prev_rewar',self.prev_reward)
            reward = reward +  self.prev_reward
            reward = [reward]
            print('sono in else')
        #print('Goal',goal)
        
            
        self.prev_reward = reward_t
        print('self.prev_reward', self.prev_reward)
        return reward, goal
        
        
        
  
        
        
