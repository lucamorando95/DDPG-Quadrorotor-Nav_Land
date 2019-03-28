#!/usr/bin/env python 

#import library ros 
import rospy 
import time
from environment import Environment
from keras.models import Sequential
import keras.backend as K
from keras import backend as K
import tensorflow as tf
#import library untuk mengirim command dan menerima data navigasi dari quadcopter
from AutonomousFlight import AutonomousFlight

from geometry_msgs.msg import Twist,PoseWithCovariance
from std_msgs.msg import String 
from std_msgs.msg import Empty 

from nav_msgs.msg import Odometry

#import class status untuk menentukan status ddari quadcopter
#from drone_status import DroneStatus

COMMAND_PERIOD = 1000
def start_training():
    env = Environment()  #Put here all teh function needed for the interaction with the env
    
    observ_dim = env.num_states
    actions_dim = env.num_actions
    
    #Define buffer size and dimension
    buffer_size = 5000
    miniBatch_size = 32
    
    #Define Hyperparameters values
    gamma = 0.98 #learning parameter --> discount factor: model the fact that future reward are worth less than immediate reward
    #MQ value factor, if settled near 1 means tha learning is quickly
    tau = 0.001# neural networks updating
    
    #training parameters
    
    explore = 10000
    max_episode = 1000
    max_steps_in_ep = 10000
    reward = 0
    done = False 
    epsilon = 0.9 #exploration exploitation value 
    indicator = 0
    
    plot_reward = False
    
    #Create Empty array for ep_reward and episode states
    episode_rewards = []
    episode = []
    
    #Say to tensorflow to run on CPU
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    
    
    
    
    
    
    
    max_ep = 25
    vel = Twist()
    i = 0
    for ep in range(4*max_ep):
        vel.linear.x = 0.5
        vel.linear.y = 0
        uav.SetCommand(vel.linear.x,vel.linear.y,0,0,0,0) 
        if (i == max_ep):
            i = 0
            t = env._reset()
            if t == True:
               print("ResetSimulation Succesfully")
            else:
               print("ResetSimulation unSuccesfully")
               break
        i = i+1
    return t
       
    
if __name__ == '__main__': 
    
    poseData = None 
    TakeOff = False
     
    try:    
        uav = AutonomousFlight()
        
        while not rospy.is_shutdown():
             if TakeOff == False:
                 takeOff = uav.SendTakeOff(TakeOff, uav)
                 TakeOff = takeOff
                 print("TakeOff",takeOff)
      
            
    #Start Training 
             else:
                 start_training()
                 print("sonoqui")
    
    except rospy.ROSInterruptException:
        pass



         
