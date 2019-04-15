#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:18:07 2019

@author: parallels
"""

import gym 
import numpy as np
from keras.models import Sequential, Model #Allow to create models layer-by-layer. Does not Permits to create models that share layers 
from keras.layers import Input, Dense, Dropout
from keras.layers.merge import Add, Multiply 
from keras.optimizers import Adam 
import keras.backend as K
import tensorflow as tf
import pdb
#from ReplayBuffer import Replay_Buffer

class ActorNetwork(object):
    
    def __init__(self, env, sess, batch_size = 32, tau = 0.120, learning_rate = 0.0001):
        self.env = env
        self.sess = sess
        
        self.observation_dim = self.env.num_states
        self.actions_dim = self.env.num_actions
        
        #network hyperparameters
        self.learning_rate = learning_rate 
        self.batch_size = batch_size 
        self.eps = 1.0
        self.eps_decay = 0.995
        self.gamma = 0.95
        self.tau = tau
        self.buffer_size = 5000
        self.hidden_layers_dim = 32
        
        #Connect to Replay Buffer 
        #self.replayBuff = Replay_Buffer(self.buffer_size)
        
        
        def initialize_actor(hidden_layer_dimension):
             #it is a vector of the same dimension of the number of states ---> 2 states
            obs_input = Input(shape = [self.observation_dim])
            #Define hidden layers of the network 
            hidden_1 = Dense(hidden_layer_dimension, activation = 'relu')(obs_input) #first layer takes the inputs value
            hidden_2 = Dense(hidden_layer_dimension, activation = 'relu')(hidden_1) #first hidden layer 
            hidden_3 = Dense(hidden_layer_dimension, activation = 'relu')(hidden_2) #second hidden layer 
            output_layer = Dense( self.actions_dim, activation = 'tanh')(hidden_3)
            #print('Output layer',output_layer)
            #Define the network model 
            model = Model(input = obs_input, output = output_layer)
            
            return model, model.trainable_weights, obs_input 
        
        
        #create network model 
        self.model, self.weights, self.state = initialize_actor(self.hidden_layers_dim)
        self.model_target, self.target_weights, self.target_state = initialize_actor(self.hidden_layers_dim)
        
        #compute gradients inside the Network
        #Use placeholder to insert values into Tensorflow from Python code 
        self.gradient_action = tf.placeholder(tf.float32, [None, self.actions_dim]) #specify the dimensionality of the tensor that must be created
        self.gradient_param = tf.gradients(self.model.output, self.weights, -self.gradient_action) #The oputput is a tensor or a lists of tensor  of the same len(self.model.output) where each element is the derivative of the weights respect to the output 
        #The gradient must be negative because we search for gradient ascend
        grads = zip(self.gradient_param, self. weights) #create a list iwhere each elemnt is an array of an elemnt from gradient param and the elemnt in the same position in weights
                                                        #[(gradient_params_1,weights_1), (gradient_params_2, weights_2)]
        
        #optimizer and run iteration for network training
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables()) #start a session where the matematical operation are evaluated and initilize all variable before starting the network tree
        
        self.writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())
        self.merge_op = tf.summary.merge_all() 
       
        
        #function that permits the actor network training 
    def actor_train(self, states, action_gradients):
        self.sess.run(self.optimize, feed_dict={self.state: states, self.gradient_action: action_gradients}) #feed_dict maps graph elements like states and gradient action to the effective values 
            
    def target_net_train(self):
        actor_weights = self.model.get_weights() #return model weights ---> flat list of numpy arrays
        actor_target_weights = self.model_target.get_weights()
            
            #update target networks
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1- self.tau) * actor_target_weights[i]
            
        self.model_target.set_weights(actor_target_weights)
       
            
        
        
        
        