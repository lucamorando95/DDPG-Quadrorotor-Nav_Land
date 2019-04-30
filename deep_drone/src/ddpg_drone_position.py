#!/usr/bin/env python 
# coding=utf-8
#import library ros 
import rospy 
import time
import numpy as np
import json
from environment import Environment
from keras.models import Sequential
from keras.models import model_from_json, Model,load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import keras.backend as K
from keras import backend as K
import tensorflow as tf
import random
import math
import os
import matplotlib.pyplot as plt #for plotting graophs

from AutonomousFlight import AutonomousFlight
from geometry_msgs.msg import Twist,PoseWithCovariance
from std_msgs.msg import String 
from std_msgs.msg import Empty 

from nav_msgs.msg import Odometry
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from ReplayBuffer import ReplayBuffer 
#import class status untuk menentukan status ddari quadcopter
#from drone_status import DroneStatus

COMMAND_PERIOD = 1000

def start_training(goal_position, test_flag):
    debug = True
    env = Environment(debug, goal_position)  #Put here all teh function needed for the interaction with the env
    
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
    max_episode = 5000
    max_steps_in_ep = 10000
    reward = 0
   
   
    done = False 
    epsilon = 0.9 #exploration exploitation value 
    indicator = 0
    
    plot_reward = False
    save_stats = True 
    #Create Empty array for Plotting VAriables
    ep_reward = []
    episode = []
    distance = []
    
    distance_step = []
    step_reward = []
    
    #Define goal pos only for print purpose 
    distance_error = []
    goal_position = [2.0, 3.0]
    episode_check = 0
    desired_checking_episode = 10
    #If running on RDS uncomment this part
    #Tensorflow GPU optimization
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    sess = tf.Session(config=config)
#    from keras import backend as K
#    K.set_session(sess)
#    
    #Say to tensorflow to run on CPU
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    #Define the actor, critic Network and Buffer
    
    actor = ActorNetwork(env,sess)
    critic = CriticNetwork(env,sess)
    replay_buffer = ReplayBuffer()    
    saved_path = '/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved' #/Model_Weights_saved'
    save_directory = os.path.join(os.getcwd(), saved_path)
    
    try:
        actor.model.load_weights("/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Actor_weights/499_actor_weights.h5")
        actor.model_target.load_weights("/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Actor_weights/499_actor_weights.h5")
        critic.model.load_weights('/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Critic_weights/499_critic_model.h5')
        critic.model_target.load_weights("/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Critic_weights/499_critic_model.h5")
        
        #critic.model_target.load_weights("/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Actor_weights/219_critic_weights.h5")
        print("WEIGHTS LOAD CORRECTLY")
    except:
        print("ERR: WEIGHTS LOAD UNCORRECTLY")
        
    
    if not os.path.isdir(save_directory): #return true if path is in an existing directory
        os.makedirs(save_directory)
    os.chdir(save_directory)
    
    #plot graphs settings
    if (plot_reward):
        plt.ion() #turn the interactive mode on
        plt.title('Training Curve')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.grid()
    
        plt.ion()
        plt.title('Distance Error')
        plt.xlabel('Episodes')
        plt.ylabel('Cartesian Error')
        plt.grid()
   #Principal Training LOOP
    for ep in range(500,max_episode):
       #receive initial observation state
       state_t = env._reset(test_flag) #reset environment ---> waiting for take off -> give also the state information relative to the actual drone position ecc 
       state_t = np.asarray(state_t) #create an array that is the state at time t : errorX,errorY, Terminal
       total_reward = [0] #initialize reward 
       terminal = [False] #flag relative to the training phase
       step = 0 #number of iteration inside eac episode 
       episode_check  = episode_check +1
       while(terminal[0] == False):
            if step > 200:#200:
                break # exit from the main loop
           
            step = step + 1
            
#            if debug:
#                print('###############################')
                #print('step: {}'.format(step)) 
            print('############################################################')
            loss = 0
            epsilon -= 1.0/explore #define the expolre exploit probabilities 
            
            action_t = np.zeros([1, actions_dim]) #create a zero array with the same dimesion of the number of actions
            noise_t = np.zeros([1, actions_dim]) #noise array
            
            #the current action is selected according to current policy and exploration noise 
            #The action is predicted from the actor network without noise 
            
            action_t_initial = actor.model.predict(state_t.reshape(1, state_t.shape[0]))#state_t.reshape(1, state_t.shape[0])) #make prediction given the state input,shape gives the dimension of the vector.
            #print('action_t_initial', action_t_initial)                                                             
            
            #adding noise to the action predicted 
            noise_t[0][0] = OUhlenbeck_noise(epsilon,action_t_initial[0][0])
            noise_t[0][1] = OUhlenbeck_noise(epsilon,action_t_initial[0][1])
            #noise_t[0][2] = OUhlenbeck_noise(epsilon,action_t_initial[0][2])
           
            action_t[0][0] = action_t_initial[0][0] + noise_t[0][0]
            action_t[0][1] = action_t_initial[0][1] + noise_t[0][1]
            
            #Step, Apply action in the environment and reach a new state 
            state_t1, reward_t, terminal, _ = env._step(action_t[0],step, test_flag) 
            #print('state_t1 : {}'.format(state_t1))
       
            state_t1 = np.asarray(state_t1) #create array of the new state
            #Now the sequence state_t, actions, reward, state_t1 must be add to the replay buffer experience 
            replay_buffer.add_experience(state_t, action_t[0], reward_t, state_t1, terminal)
            
           
            
            
            #Sample a new experience (set of sate, action, state1, reward, terminal) from batch 
            mini_batch = replay_buffer.take_experience()
            
            states_buff = np.asarray([i[0] for i in mini_batch])
            actions_buff = np.asarray([i[1] for i in mini_batch])
            reward_buff = np.asarray([i[2] for i in mini_batch])
            state_new_buff = np.asarray([i[3] for i in mini_batch])
            terminal_buff = np.asarray([i[4] for i in mini_batch])
                #istantiate a y_target vector which must be of the same dimesion of the length of the mini batch
            #y_target = np.asarray([i[1] for i in mini_batch]) #it is only to have the array of the desired dimension
           
            #Predic an action from Actor Network given the state_new_buff from mini_batch
            action_new_buff = actor.model_target.predict(state_new_buff)
            
            #Take the prediction from the Critic network about possible Q target relatives to the new_state and action taken from mini batch
            Q_target_predicted = critic.model_target.predict([state_new_buff, action_new_buff])
#            print('Q_target_predicted', Q_target_predicted)
#            print('reward_buff', reward_buff)
            #Update the target of the Q value evaluating the BElmann Equation
            y_target = []
            for j in range(len(mini_batch)):
               
                if terminal_buff[j]:
                   #y_target[j] =  reward_buff[j]
                   y_target.append(reward_buff[j])
                else:
                   
                   y_target.append(reward_buff[j] + gamma* Q_target_predicted[j])  #it append every time an array and create a sort of list
           
            #i resize all in order to obtain an array with 1 column and many rows as the dimension of the batch       
            y_target = np.resize(y_target,[len(mini_batch),1]) 
           
            #Evaluate the loss error utilizing the model.train_on_batch and update the weights of the critic
            #having as target the y_target evaluated from the belmann equation 
            loss = loss + critic.model.train_on_batch([states_buff, actions_buff], y_target) # L = 1/N * sum(y_target - Q(si,ai|theta^Q)^2)
            
            #The actor policy is updated using the sampled policy gradient 
            ############ see https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html for the full expalantion
            #An action is predicted taking the states from the buffer. The action predicted will be used to evaluate the increasing of the critic_gradient
            action_for_grad = actor.model.predict(states_buff)
            #The actor network is trained computing the gradient of the critic network repect to actions. 
            #This because the actor network must be trained to follow the maximum gradient increasing direction of the critic network that represent in fact the q network.
            #Like in Q learning, in the Q table, you follow tha action that increase the Q value. Sa me choose here, only different is that instead of having a value 
            #to follow we have the gradient of the Critic NEtwork
            critic_gradient = critic.gradients(states_buff,action_for_grad)
            #The actor network is trained having as input the states from which the critic gradient is computed and as target the critic_gradient itself. 
            #The goal of the actor networ is to output actions that goes in the direction of the gradient and every time maximize it
            actor.actor_train(states_buff,critic_gradient)
            #The last two rows are done in order to updates the target network
            #theta^Q = tau*theta^Q +(1- tau)*theta^Q'
            actor.target_net_train()
            critic.target_net_train()
            
            #Evaluate distance error fro print purpose 
            error_x = (goal_position[0] - state_t[0])
            error_y = (goal_position[1] - state_t[1])
            distance_error = math.sqrt(error_x*error_x + error_y*error_y)
            
            #Update Total Reward 
            #print('reward_t', reward_t)
           
            if not reward_t[0] :
                reward_t[0] = -100*distance_error
                
            total_reward[0] =  total_reward[0] + reward_t[0]
           
            #The new state becomes the actual state
            state_t = state_t1
            
            #### Save distance and reward for each step only for pllotting purpose
            distance_step.append(distance_error)
            step_reward.append(reward_t[0])
            if (terminal[0] == True or step == 200):
                   distance_step_mat = np.asarray(distance_step)
                  
                   step_reward_mat = np.asarray(step_reward)
                   
                   
                  
                   distance_step_name = 'Statistics/Step_Statistics/%d_distance_step.csv' %(ep)
                   step_reward_name = 'Statistics/Step_Statistics/%d_step_reward.csv' %(ep)
                   
                   np.savetxt(distance_step_name,distance_step_mat, delimiter = ",") #Nel post processing in matlab importare il vettore episode su asse x e fare plot con reward e distance su asse y
                   np.savetxt(step_reward_name,step_reward_mat, delimiter = ",")
                   distance_step_mat = []
                   step_reward_mat = []
                   distance_step = []
                   step_reward = []
            
            #Save Model and Weights every 50 episodes as a checkpoint 
            print('episode: {}, steps: {}, tot_rewards: {}, terminal: {}'.format(ep, step, total_reward, terminal))
            
            print('distance_error:{}, pos_x: {}, pos_y: {}'.format(distance_error, state_t[0], state_t[1]))
            
            #if ((step+1)%10 == 0):
       if (episode_check == desired_checking_episode):
                #save Model 
              action_model_name = 'Actor_weights/%d_actor_model.h5' %(ep)
              critic_model_name = 'Critic_weights/%d_critic_model.h5' %(ep)
              save_path = os.path.join(save_directory, action_model_name)
              actor.model.save(action_model_name) #True if you want to overwrite
              critic.model.save(critic_model_name) 
              print('Model Saved in path: %s' %save_directory)
                
                #Save Weights
              model_ext = ".h5"
              model_ext2 = ".json"
              action_save_weights_name = 'Actor_weights/%d_actor_weights' %(ep)
              actor.model.save_weights(action_save_weights_name+model_ext,overwrite = True)  #Save Weights
              with open(action_save_weights_name+model_ext2, "w") as outfile:
                  json.dump(actor.model.to_json(), outfile) #save Model Archutecture, not weights
                
                
              critic_save_weights_name = 'critic_weights/ %d_critic_weights' %(ep)
              critic.model.save_weights(critic_save_weights_name+model_ext,overwrite = True)
              with open(critic_save_weights_name+model_ext2, "w") as outfile:
                  json.dump(critic.model.to_json(), outfile)
                
              print('Weights Saved in path: %s' %save_directory)
              
         #######################
            #Save Statistics 
       if (save_stats):
              
               episode.append(ep)
               ep_reward.append(total_reward[0])
               distance.append(distance_error)
               
               if (episode_check == desired_checking_episode):
                   
                   ep_reward_mat = np.asarray(ep_reward)
                   episode_mat = np.asarray([episode])
                   distance_mat = np.asarray(distance)
                   
                   episode_mat = np.resize(episode_mat,[ep,1])
                  
                   episode_name = 'Statistics/%d_episode.csv' %(ep)
                   episode_reward_name = 'Statistics/%d_reward.csv' %(ep)
                   distance_name = 'Statistics/%d_distance.csv' %(ep)
                   np.savetxt(episode_name,episode_mat, delimiter = ",") #Nel post processing in matlab importare il vettore episode su asse x e fare plot con reward e distance su asse y
                   np.savetxt(episode_reward_name,ep_reward_mat, delimiter = ",")
                   np.savetxt(distance_name,distance_mat, delimiter = ",")
                   episode_check = 0
         
            ##################################
            #Plot State every tot episode 
            
            #Da fare quando si arriva alla fase di test
            
            
            ######################################################
            #Plotting REwards and Distance Error
#            if (plot_reward):
#                ep_reward.append(total_reward)
#                episode.append(step)
#                distance.append(distance_error)
#                if (step_check == desired_checking_step):
#                   plt.plot(episode, ep_reward)
#                   plt.plot(episode, distance)
#                   plt.pause(0.001)
            
           
      # plt.savefig("Training Curve.png")
       #plt.savefig("Distance Error.png")
       #return reward_t
       
    


def OUhlenbeck_noise(epsilon, action):
    mu = 0.0
    theta = 0.60
    sigma = 0.3
    
    dx = max(epsilon,0) *(theta * (mu - action) + sigma*np.random.randn(1))
    return dx 

def start_test(goal_position, test_flag):
    
    """
    During the test phase the agents use only the action that are predicted from the actor network, well trained.
    In practice every step an action is predicted anbd a reward is done for that action.
    If the network is well trained is must be able every time to maximize the reward in 
    order to reach the goal 
    """
    
    
    debug = True
    env = Environment(debug, goal_position)  #Put here all teh function needed for the interaction with the env
    
    observ_dim = env.num_states
    actions_dim = env.num_actions
    #Define Hyperparameters values
    gamma = 0.98 #learning parameter --> discount factor: model the fact that future reward are worth less than immediate reward
    #MQ value factor, if settled near 1 means tha learning is quickly
    tau = 0.001# neural networks updating
    
    #goal_position = [5.0, -3.0]
    
    max_episode = 30
    max_steps = 1000
    reward = 0
    terminal = [False]
    save_stats = True
     #Say to tensorflow to run on CPU
    config = tf.ConfigProto(device_count = {'GPU': 0})
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    #write directory where find model saved 
    load_path = '/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved' #/Model_Weights_saved'
    load_directory = os.path.join(os.getcwd(), load_path)
    
    if not os.path.isdir(load_directory): #return true if path is in an existing directory
        os.makedirs(load_directory)
    os.chdir(load_directory)
    
    mean_reward = []
    std_reward = []
    ep_reward = []
    episode = []
    distance = []
    #Load Model to test every 30 episode.
    #Load Model 200, 230 ecc to 500
    for i in range(299, 499, 20):
        print(i)
        #Load actor and critic model
#        actor_model = '/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Actor_weights/%d_actor_model.h5' %(i)
#        critic_model = '/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Critic_weights/%d_critic_model.h5' %(i)
#        
#        path_actor = os.path.join(load_directory, actor_model)
#        path_critic = os.path.join(load_directory, critic_model)
#        print(path_actor)
#        try:
#            actor = load_model(actor_model)
#            critic = load_model(critic_model)
#            print('actor', actor)
#            print('Model weight succesfully')
#        except:
#            print('ERROR: Model weight not succesfully') 
        
        actor = ActorNetwork(env,sess)
        #critic = CriticNetwork(env,sess)
        #actor_model = '/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Actor_weights/%d_actor_model.h5' %(i) QUESTO È QUELLO ORIGINALE 
        actor_model = '/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Actor_weights/499_actor_model.h5'
       # critic_model = '/home/parallels/catkin_ws/src/deep_drone/src/Data_Saved/Critic_weights/%d_critic_model.h5' %(i)
        try:
             actor.model.load_weights(actor_model)
             actor.model_target.load_weights(actor_model)
             #critic.model.load_weights(critic_model)
             #critic.model_target.load_weights(critic_model)
        
        
             print("WEIGHTS LOAD CORRECTLY")
        except:
             print("ERR: WEIGHTS LOAD UNCORRECTLY")
    
    
        episode_reward = []
        #model_num.append(i)
        episode_check = 0
        desired_checking_episode = 20
        
        for ep in range(max_episode):
            
            #Take initial observation as in training phase 
            state_t = env._reset(test_flag) #reset environment ---> waiting for take off -> give also the state information relative to the actual drone position ecc 
            state_t = np.asarray(state_t) #create an array that is the state at time t : errorX,errorY, Terminal
            total_reward = [0] #initialize reward 
            terminal = [False] #flag relative to the training phase
            step = 0 #number of iteration inside eac episode 
            episode_check  = episode_check +1
            
            while(terminal[0] == False):
                if step > 200:
                    break
                print('############################################################')
                step = step + 1
                
                action_t = np.zeros([1, actions_dim]) #create a zero array with the same dimesion of the number of actions
               
                action_t_initial = actor.model.predict(state_t.reshape(1, state_t.shape[0]))
                
                action_t[0][0] = action_t_initial[0][0]
                action_t[0][1] = action_t_initial[0][1]
                
                #Step, Apply action in the environment and reach a new state 
                state_t1, reward_t, terminal, altitude = env._step(action_t[0],step, test_flag) 
                state_t1 = np.asarray(state_t1)
                
                total_reward[0] =  total_reward[0] + reward_t[0]
                state_t = state_t1
                
                #Evaluate distance error fro print purpose 
                error_x = (goal_position[0] - state_t[0])
                error_y = (goal_position[1] - state_t[1])
                distance_error = math.sqrt(error_x*error_x + error_y*error_y)
                if terminal == [True] and distance_error < 1.5:#0.7
                    uav = AutonomousFlight()
                    print("waiting For Landing")
                    landing = False 
                    
                    while not rospy.is_shutdown():
                        if landing == False:
                            poseData, _, _, altitudeVelDrone = env.takeEnvObservations(test_flag)
                            landing = uav.SendLand(uav,poseData,altitudeVelDrone)
                        else:
                            break
                        rospy.sleep(0.5)
                    
                print('episode: {}, step: {},distance_error: {}, total_reward :{}'.format(ep, step,distance_error, total_reward[0]))
            episode_reward.append(total_reward)
            print('episode: {}, total_ep_reward :{}'.format(ep, np.mean(episode_reward)))
            if (save_stats):
              
               episode.append(ep)
               mean_reward.append(np.mean(episode_reward))
               std_reward.append(np.std(episode_reward))
               ep_reward.append(total_reward[0])
               distance.append(distance_error)
               
               if (episode_check == desired_checking_episode):
                   
                   ep_reward_mat = np.asarray(ep_reward)
                   episode_mat = np.asarray([episode])
                   distance_mat = np.asarray(distance)
                   mean_reward_mat = np.asarray(mean_reward)
                   std_reward_mat = np.asarray(std_reward)
                   episode_mat = np.resize(episode_mat,[ep,1])
                   
                   
                   episode_name = load_path +'/Test_Statistics/%d_test_episode.csv' %(ep)
                   episode_reward_name = load_path +'/Test_Statistics/%d_test_reward.csv' %(ep)
                   distance_name = load_path +'/Test_Statistics/%d_test_distance.csv' %(ep)
                   mean_reward_name = load_path + '/Test_Statistics/%d_test_mean_reward.csv' %(ep)
                   std_reward_name = load_path + '/Test_Statistics/%d_test_std_reward.csv'  %(ep)
                   np.savetxt(episode_name,episode_mat, delimiter = ",") #Nel post processing in matlab importare il vettore episode su asse x e fare plot con reward e distance su asse y
                   np.savetxt(episode_reward_name,ep_reward_mat, delimiter = ",")
                   np.savetxt(mean_reward_name,mean_reward_mat, delimiter = ",")
                   np.savetxt(std_reward_name,std_reward_mat, delimiter = ",")
                   np.savetxt(distance_name,distance_mat, delimiter = ",")
                   
                   print('Statistics saved succesfully in directory:', load_path, '/Test_Statistics/' )
                   episode_check = 0
    
    
if __name__ == '__main__': 
    
    poseData = None 
    TakeOff = False
    test_flag = True
    goal_position = [2.0, 3.0]
    try:    
        if test_flag:
              start_test(goal_position,test_flag )
    #Start Training 
        else:
              start_training(goal_position, test_flag)
   
    except rospy.ROSInterruptException:
        pass



         
