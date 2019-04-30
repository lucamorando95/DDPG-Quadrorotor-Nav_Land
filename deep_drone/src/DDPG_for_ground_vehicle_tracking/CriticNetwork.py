#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:56:48 2019

@author: parallels
"""

import gym 
import numpy as np
import math
from keras.models import model_from_json, load_model, save_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.models import Model
from keras.optimizers import Adam 
import keras.backend as K
import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, env, sess, batch_size = 32, tau = 0.120, learning_rate = 0.001):
        self.env = env 
        self.sess = sess
        self.batch_size = batch_size
        
        self.observation_dim = self.env.num_states
        self.actions_dim = self.env.num_actions
        
        #Network Hyperparamter 
        self.learning_rate = learning_rate 
        self.batch_size = batch_size 
        self.tau = tau
        self.buffer_size = 5000
        self.hidden_layers_dim = 32
        
        K.set_session(sess) #set the global tensorflow session
        
        
        def initialize_critic_net(hidden_layers_dimension):
        #The critic network as two input that are the state and the actions.
        #The state enter in the input layer, then ther'is  an hodden layer to process the state in input.
        #The actions that are the second input enters in the second layer of the network 
        
        #INPUT 1 ---> States
        
           state_input = Input(shape = [self.observation_dim])
           input_layer = Dense(hidden_layers_dimension, activation = 'relu')(state_input) #input layer --> relu takes only the positive weights, the negative ones are conducted to zero
           hidden_1 = Dense(hidden_layers_dimension, activation = 'linear')(input_layer) #Identity activation function --> maintains inaltereted the weights values
        
           #INPUT 2 ---> Actions 
         
           action_input =  Input(shape = [self.actions_dim], name = 'action_in')
           action_layer = Dense(hidden_layers_dimension,activation = 'linear')(action_input)
        
           #Merge the states and action layer in one singular networks
           hidden_2 = merge([hidden_1,action_layer], mode = 'sum')
           hidden_3 = Dense(hidden_layers_dimension,activation = 'relu')(hidden_2)
           out_layer = Dense(1,activation = 'relu')(hidden_3)
           
         
           model = Model(input = [state_input,action_input], output = [out_layer])
           adam_opt = Adam(self.learning_rate)
           model.compile(loss = 'mse', optimizer = adam_opt)
        
           return model, state_input, action_input
        
        
        
        
        #create network model 
        self.model, self.state, self.action = initialize_critic_net(self.hidden_layers_dim)
        self.model_target, self.target_action, self.target_state = initialize_critic_net(self.hidden_layers_dim)
        self.gradient_action = tf.gradients(self.model.output, self.action) #The oputput is a tensor or a lists of tensor  of the same len(self.model.output) where each element is the derivative of the weights respect to the output
        self.sess.run(tf.initialize_all_variables())
        
   
        
    def gradients(self, states, actions):
       
        return self.sess.run(self.gradient_action, feed_dict = {self.state:states, self.action:actions})[0]  #Da verificare 
    
    
    def target_net_train(self):
        critic_weights = self.model.get_weights() #returns model weights as a flat list of numpy array
        critic_target_weights = self.model_target.get_weights()
        
        #Update target netwoks in a slow way 
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1-self.tau)*critic_target_weights[i]
        
        self.model_target.set_weights(critic_target_weights)
        
        
        
        
        
        
        
        
        