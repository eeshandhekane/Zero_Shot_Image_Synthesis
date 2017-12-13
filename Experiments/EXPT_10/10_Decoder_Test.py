# Dependencies
import tensorflow as tf
import numpy as np
import cv2
import os, sys, re
import time
import random


# Load MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)


# Parameters
IMG_SIZE = 28*28 # MNIST Image Size, Linearized
CAPSULE_NUM = 7 # Number of "small" attributes
LATENT_DIM = 5 # Number of "hidden" noise characteristics
ENC_DIM = 75 # First layer of inputs size
DEC_DIM = 75 # First layer of outputs size
TRAINING_ITR = 15000 + 1
RECORD_ITR = 1000
DISPLAY_ITR = 10
PKEEP = 0.7
LR = 0.001
VAE_LR = 0.001
attribute_LR = 0.001
BATCH_SIZE = 32


# Data loader class
class dataLoader(object) :
	"""
	The class to load numbers and get the attribute representations as well!!
	"""
	def __init__(self) :
		self.mnist = input_data.read_data_sets('/tmp/data', one_hot = True)


	# Next general batch
	def GetNextBatch(self, batch_size = BATCH_SIZE) :
		batch_X_, batch_Y__ = self.mnist.train.next_batch(batch_size)
		batch_Y_ = np.argmax(batch_Y__, axis = 1)
		batch_attribute_ = []
		for i in range(batch_size) :
			if batch_Y_[i] == 0 : 
				batch_attribute_.append([1, 1, 1, 0, 1, 1, 1])
			if batch_Y_[i] == 1 : 
				batch_attribute_.append([0, 0, 1, 0, 0, 1, 0])
			if batch_Y_[i] == 2 : 
				batch_attribute_.append([1, 0, 1, 1, 1, 0, 1])	
			if batch_Y_[i] == 3 : 
				batch_attribute_.append([1, 0, 1, 1, 0, 1, 1])		
			if batch_Y_[i] == 4 : 
				batch_attribute_.append([0, 1, 1, 1, 0, 1, 0])
			if batch_Y_[i] == 5 : 
				batch_attribute_.append([1, 1, 0, 1, 0, 1, 1])
			if batch_Y_[i] == 6 : 
				batch_attribute_.append([1, 1, 0, 1, 1, 1, 1])		
			if batch_Y_[i] == 7 : 
				batch_attribute_.append([1, 0, 1, 0, 0, 1, 0])			
			if batch_Y_[i] == 8 :
				batch_attribute_.append([1, 1, 1, 1, 1, 1, 1])			
			if batch_Y_[i] == 9 :
				batch_attribute_.append([1, 1, 1, 1, 0, 1, 0 ])
		batch_attribute_np = np.array(batch_attribute_).astype(np.float32)
		return batch_X_, batch_Y_, batch_Y__, batch_attribute_np


	# Define a function to get a batch of permissible digits (source class images)
	def GetNextPermBatchOfRandomSize(self, permissible_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) :
		# Get a random permissible number
		this_batch_int = random.choice(permissible_list)
		while 1 :			
			batch_X_, batch_Y__ = self.mnist.train.next_batch(100)
			batch_Y_ = list(np.argmax(batch_Y__, axis = 1))
			indices_this_batch_int = [i for i, j in enumerate(batch_Y_) if j == this_batch_int]
			l = len(indices_this_batch_int)
			if len(indices_this_batch_int) > 1 :
				if this_batch_int == 0 :
					attr_0 = np.ones([l, 1]).astype(np.float32)
					attr_1 = np.ones([l, 1]).astype(np.float32)
					attr_2 = np.ones([l, 1]).astype(np.float32)
					attr_3 = np.zeros([l, 1]).astype(np.float32)
					attr_4 = np.ones([l, 1]).astype(np.float32)
					attr_5 = np.ones([l, 1]).astype(np.float32)
					attr_6 = np.ones([l, 1]).astype(np.float32)
				if this_batch_int == 1 :
					attr_0 = np.zeros([l, 1]).astype(np.float32)
					attr_1 = np.zeros([l, 1]).astype(np.float32)
					attr_2 = np.ones([l, 1]).astype(np.float32)
					attr_3 = np.zeros([l, 1]).astype(np.float32)
					attr_4 = np.zeros([l, 1]).astype(np.float32)
					attr_5 = np.ones([l, 1]).astype(np.float32)
					attr_6 = np.zeros([l, 1]).astype(np.float32)
				if this_batch_int == 2 :
					attr_0 = np.ones([l, 1]).astype(np.float32)
					attr_1 = np.zeros([l, 1]).astype(np.float32)
					attr_2 = np.ones([l, 1]).astype(np.float32)
					attr_3 = np.ones([l, 1]).astype(np.float32)
					attr_4 = np.ones([l, 1]).astype(np.float32)
					attr_5 = np.zeros([l, 1]).astype(np.float32)
					attr_6 = np.ones([l, 1]).astype(np.float32)
				if this_batch_int == 3 :
					attr_0 = np.ones([l, 1]).astype(np.float32)
					attr_1 = np.zeros([l, 1]).astype(np.float32)
					attr_2 = np.ones([l, 1]).astype(np.float32)
					attr_3 = np.ones([l, 1]).astype(np.float32)
					attr_4 = np.zeros([l, 1]).astype(np.float32)
					attr_5 = np.ones([l, 1]).astype(np.float32)
					attr_6 = np.ones([l, 1]).astype(np.float32)
				if this_batch_int == 4 :
					attr_0 = np.zeros([l, 1]).astype(np.float32)
					attr_1 = np.ones([l, 1]).astype(np.float32)
					attr_2 = np.ones([l, 1]).astype(np.float32)
					attr_3 = np.ones([l, 1]).astype(np.float32)
					attr_4 = np.zeros([l, 1]).astype(np.float32)
					attr_5 = np.ones([l, 1]).astype(np.float32)
					attr_6 = np.zeros([l, 1]).astype(np.float32)
				if this_batch_int == 5 :
					attr_0 = np.ones([l, 1]).astype(np.float32)
					attr_1 = np.ones([l, 1]).astype(np.float32)
					attr_2 = np.zeros([l, 1]).astype(np.float32)
					attr_3 = np.ones([l, 1]).astype(np.float32)
					attr_4 = np.zeros([l, 1]).astype(np.float32)
					attr_5 = np.ones([l, 1]).astype(np.float32)
					attr_6 = np.ones([l, 1]).astype(np.float32)
				if this_batch_int == 6 :
					attr_0 = np.ones([l, 1]).astype(np.float32)
					attr_1 = np.ones([l, 1]).astype(np.float32)
					attr_2 = np.zeros([l, 1]).astype(np.float32)
					attr_3 = np.ones([l, 1]).astype(np.float32)
					attr_4 = np.ones([l, 1]).astype(np.float32)
					attr_5 = np.ones([l, 1]).astype(np.float32)
					attr_6 = np.ones([l, 1]).astype(np.float32)
				if this_batch_int == 7 :
					attr_0 = np.ones([l, 1]).astype(np.float32)
					attr_1 = np.zeros([l, 1]).astype(np.float32)
					attr_2 = np.ones([l, 1]).astype(np.float32)
					attr_3 = np.zeros([l, 1]).astype(np.float32)
					attr_4 = np.zeros([l, 1]).astype(np.float32)
					attr_5 = np.ones([l, 1]).astype(np.float32)
					attr_6 = np.zeros([l, 1]).astype(np.float32)
				if this_batch_int == 8 :
					attr_0 = np.ones([l, 1]).astype(np.float32)
					attr_1 = np.ones([l, 1]).astype(np.float32)
					attr_2 = np.ones([l, 1]).astype(np.float32)
					attr_3 = np.ones([l, 1]).astype(np.float32)
					attr_4 = np.ones([l, 1]).astype(np.float32)
					attr_5 = np.ones([l, 1]).astype(np.float32)
					attr_6 = np.ones([l, 1]).astype(np.float32)
				if this_batch_int == 9 :
					attr_0 = np.ones([l, 1]).astype(np.float32)
					attr_1 = np.ones([l, 1]).astype(np.float32)
					attr_2 = np.ones([l, 1]).astype(np.float32)
					attr_3 = np.ones([l, 1]).astype(np.float32)
					attr_4 = np.zeros([l, 1]).astype(np.float32)
					attr_5 = np.ones([l, 1]).astype(np.float32)
					attr_6 = np.zeros([l, 1]).astype(np.float32)
				batch_ = np.array([batch_X_[i] for i in indices_this_batch_int])
				return this_batch_int, batch_, attr_0, attr_1, attr_2, attr_3, attr_4, attr_5, attr_6


# Reset the graph
tf.reset_default_graph()


# Instantiate a data loader
data_loader = dataLoader()
# trial_batch_X_, trial_batch_Y_, trial_batch_Y__, trial_batch_attribute_np = data_loader.GetNextBatch(5)
# print trial_batch_Y__, '\n', trial_batch_Y_, '\n', trial_batch_attribute_np, '\n'
trial_this_batch_int, trial_next_perm_batch_of_random_size, trial_a0, trial_a1, trial_a2, trial_a3, trial_a4, trial_a5, trial_a6 = data_loader.GetNextPermBatchOfRandomSize(permissible_list = [0, 1, 2, 3, 4, 5])
print trial_this_batch_int, '\n', trial_next_perm_batch_of_random_size.shape, '\n', trial_a0.shape, '\n', trial_a1.shape, '\n', trial_a2.shape, '\n', trial_a3.shape, '\n', trial_a4.shape, '\n', trial_a5.shape, '\n', trial_a6.shape, '\n', np.all(trial_next_perm_batch_of_random_size == 0), '\n', trial_a0
# sys.exit()


# Define placeholders and variables
X = tf.placeholder(tf.float32, shape = [None, IMG_SIZE], name = 'X')


# 0th Capsule Parameters
ph_A0 = tf.placeholder(tf.float32, shape = [None, 1], name = 'ph_A0')
test_noise_A0 = tf.placeholder(tf.float32, shape = [None, LATENT_DIM], name = 'test_noise_A0')
# ENC_W1_A0 = tf.get_variable(shape = [IMG_SIZE, ENC_DIM],  name = 'ENC_W1_A0')
# ENC_B1_A0 = tf.get_variable(shape = [ENC_DIM],  name = 'ENC_B1_A0')
# ENC_W2_A0 = tf.get_variable(shape = [ENC_DIM, LATENT_DIM],  name = 'ENC_W2_A0')
# ENC_B2_A0 = tf.get_variable(shape = [LATENT_DIM],  name = 'ENC_B2_A0')
# SAM_W_mu_A0 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_mu_A0')
# SAM_B_mu_A0 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_mu_A0')
# SAM_W_logstd_A0 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_logstd_A0')
# SAM_B_logstd_A0 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_logstd_A0')
DEC_W1_A0 = tf.get_variable(shape = [LATENT_DIM, ENC_DIM],  name = 'DEC_W1_A0')
DEC_B1_A0 = tf.get_variable(shape = [ENC_DIM],  name = 'DEC_B1_A0')
DEC_W2_A0 = tf.get_variable(shape = [ENC_DIM, IMG_SIZE],  name = 'DEC_W2_A0')
DEC_B2_A0 = tf.get_variable(shape = [IMG_SIZE],  name = 'DEC_B2_A0')
ATT_W1_A0 = tf.get_variable(shape = [ENC_DIM, 1],  name = 'ATT_W1_A0')
ATT_B1_A0 = tf.get_variable(shape = [1],  name = 'ATT_B1_A0')
# 0th Capsule Forward Pass
# Y_1_A0 = tf.add(tf.matmul(X, ENC_W1_A0), ENC_B1_A0)
# Y_2_A0 = tf.nn.relu(Y_1_A0)
# Y_2_1_A0 = tf.nn.dropout(Y_2_A0, PKEEP)
# Y_3_A0 = tf.add(tf.matmul(Y_2_1_A0, ENC_W2_A0), ENC_B2_A0)
# attr_1_A0 = tf.add(tf.matmul(Y_2_1_A0, ATT_W1_A0), ATT_B1_A0)
# attr_A0 = tf.nn.sigmoid(attr_1_A0)
# Y_4_A0 = tf.nn.tanh(Y_3_A0)
# Y_4_1_A0 = tf.nn.dropout(Y_4_A0, PKEEP)
# mu_A0 = tf.add(tf.matmul(Y_4_1_A0, SAM_W_mu_A0), SAM_B_mu_A0)
# logstd_A0 = tf.add(tf.matmul(Y_4_1_A0, SAM_W_logstd_A0), SAM_B_logstd_A0)
# noise_A0 = tf.random_normal([1, LATENT_DIM])
# Z_A0 = mu_A0 + tf.multiply(noise_A0, tf.exp(.5*logstd_A0))
Y_5_A0 = tf.add(tf.matmul(test_noise_A0, DEC_W1_A0), DEC_B1_A0)
Y_6_A0 = tf.nn.relu(Y_5_A0)
Y_6_1_A0 = tf.nn.dropout(Y_6_A0, PKEEP)
Y_7_A0 = tf.add(tf.matmul(Y_6_1_A0, DEC_W2_A0), DEC_B2_A0)
Y_7_1_A0 = tf.multiply(Y_7_A0, ph_A0)
Y_rec_A0 = tf.nn.sigmoid(Y_7_1_A0)
# 0th Capsule KLT Loss
# KLT_A0 = -0.5*tf.reduce_sum(1 + 2*logstd_A0 - tf.pow(mu_A0, 2) - tf.exp(2*logstd_A0), axis = 1)
# 0th Capsule Attr Pred
# attr_loss_A0 = tf.nn.sigmoid_cross_entropy_with_logits(logits = attr_A0, labels = ph_A0)


# 1st Capsule Parameters
ph_A1 = tf.placeholder(tf.float32, shape = [None, 1], name = 'ph_A1')
test_noise_A1 = tf.placeholder(tf.float32, shape = [None, LATENT_DIM], name = 'test_noise_A1')
# ENC_W1_A1 = tf.get_variable(shape = [IMG_SIZE, ENC_DIM],  name = 'ENC_W1_A1')
# ENC_B1_A1 = tf.get_variable(shape = [ENC_DIM],  name = 'ENC_B1_A1')
# ENC_W2_A1 = tf.get_variable(shape = [ENC_DIM, LATENT_DIM],  name = 'ENC_W2_A1')
# ENC_B2_A1 = tf.get_variable(shape = [LATENT_DIM],  name = 'ENC_B2_A1')
# SAM_W_mu_A1 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_mu_A1')
# SAM_B_mu_A1 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_mu_A1')
# SAM_W_logstd_A1 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_logstd_A1')
# SAM_B_logstd_A1 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_logstd_A1')
DEC_W1_A1 = tf.get_variable(shape = [LATENT_DIM, ENC_DIM],  name = 'DEC_W1_A1')
DEC_B1_A1 = tf.get_variable(shape = [ENC_DIM],  name = 'DEC_B1_A1')
DEC_W2_A1 = tf.get_variable(shape = [ENC_DIM, IMG_SIZE],  name = 'DEC_W2_A1')
DEC_B2_A1 = tf.get_variable(shape = [IMG_SIZE],  name = 'DEC_B2_A1')
ATT_W1_A1 = tf.get_variable(shape = [ENC_DIM, 1],  name = 'ATT_W1_A1')
ATT_B1_A1 = tf.get_variable(shape = [1],  name = 'ATT_B1_A1')
# 1st Capsule Forward Pass
# Y_1_A1 = tf.add(tf.matmul(X, ENC_W1_A1), ENC_B1_A1)
# Y_2_A1 = tf.nn.relu(Y_1_A1)
# Y_2_1_A1 = tf.nn.dropout(Y_2_A1, PKEEP)
# Y_3_A1 = tf.add(tf.matmul(Y_2_1_A1, ENC_W2_A1), ENC_B2_A1)
# Y_4_A1 = tf.nn.tanh(Y_3_A1)
# attr_1_A1 = tf.add(tf.matmul(Y_2_1_A1, ATT_W1_A1), ATT_B1_A1)
# attr_A1 = tf.nn.sigmoid(attr_1_A1)
# Y_4_1_A1 = tf.nn.dropout(Y_4_A1, PKEEP)
# mu_A1 = tf.add(tf.matmul(Y_4_1_A1, SAM_W_mu_A1), SAM_B_mu_A1)
# logstd_A1 = tf.add(tf.matmul(Y_4_1_A1, SAM_W_logstd_A1), SAM_B_logstd_A1)
# noise_A1 = tf.random_normal([1, LATENT_DIM])
# Z_A1 = mu_A1 + tf.multiply(noise_A1, tf.exp(.5*logstd_A1))
Y_5_A1 = tf.add(tf.matmul(test_noise_A1, DEC_W1_A1), DEC_B1_A1)
Y_6_A1 = tf.nn.relu(Y_5_A1)
Y_6_1_A1 = tf.nn.dropout(Y_6_A1, PKEEP)
Y_7_A1 = tf.add(tf.matmul(Y_6_1_A1, DEC_W2_A1), DEC_B2_A1)
Y_7_1_A1 = tf.multiply(Y_7_A1, ph_A1)
Y_rec_A1 = tf.nn.sigmoid(Y_7_1_A1)
# 1st Capsule KLT Loss
# KLT_A1 = -0.5*tf.reduce_sum(1 + 2*logstd_A1 - tf.pow(mu_A1, 2) - tf.exp(2*logstd_A1), axis = 1)
# 0th Capsule Attr Pred
# attr_loss_A1 = tf.nn.sigmoid_cross_entropy_with_logits(logits = attr_A1, labels = ph_A1)


# 0th Capsule Parameters
ph_A2 = tf.placeholder(tf.float32, shape = [None, 1], name = 'ph_A2')
test_noise_A2 = tf.placeholder(tf.float32, shape = [None, LATENT_DIM], name = 'test_noise_A2')
# ENC_W1_A2 = tf.get_variable(shape = [IMG_SIZE, ENC_DIM],  name = 'ENC_W1_A2')
# ENC_B1_A2 = tf.get_variable(shape = [ENC_DIM],  name = 'ENC_B1_A2')
# ENC_W2_A2 = tf.get_variable(shape = [ENC_DIM, LATENT_DIM],  name = 'ENC_W2_A2')
# ENC_B2_A2 = tf.get_variable(shape = [LATENT_DIM],  name = 'ENC_B2_A2')
# SAM_W_mu_A2 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_mu_A2')
# SAM_B_mu_A2 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_mu_A2')
# SAM_W_logstd_A2 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_logstd_A2')
# SAM_B_logstd_A2 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_logstd_A2')
DEC_W1_A2 = tf.get_variable(shape = [LATENT_DIM, ENC_DIM],  name = 'DEC_W1_A2')
DEC_B1_A2 = tf.get_variable(shape = [ENC_DIM],  name = 'DEC_B1_A2')
DEC_W2_A2 = tf.get_variable(shape = [ENC_DIM, IMG_SIZE],  name = 'DEC_W2_A2')
DEC_B2_A2 = tf.get_variable(shape = [IMG_SIZE],  name = 'DEC_B2_A2')
ATT_W1_A2 = tf.get_variable(shape = [ENC_DIM, 1],  name = 'ATT_W1_A2')
ATT_B1_A2 = tf.get_variable(shape = [1],  name = 'ATT_B1_A2')
# 0th Capsule Forward Pass
# Y_1_A2 = tf.add(tf.matmul(X, ENC_W1_A2), ENC_B1_A2)
# Y_2_A2 = tf.nn.relu(Y_1_A2)
# Y_2_1_A2 = tf.nn.dropout(Y_2_A2, PKEEP)
# Y_3_A2 = tf.add(tf.matmul(Y_2_1_A2, ENC_W2_A2), ENC_B2_A2)
# Y_4_A2 = tf.nn.tanh(Y_3_A2)
# attr_1_A2 = tf.add(tf.matmul(Y_2_1_A2, ATT_W1_A2), ATT_B1_A2)
# attr_A2 = tf.nn.sigmoid(attr_1_A2)
# Y_4_1_A2 = tf.nn.dropout(Y_4_A2, PKEEP)
# mu_A2 = tf.add(tf.matmul(Y_4_1_A2, SAM_W_mu_A2), SAM_B_mu_A2)
# logstd_A2 = tf.add(tf.matmul(Y_4_1_A2, SAM_W_logstd_A2), SAM_B_logstd_A2)
# noise_A2 = tf.random_normal([1, LATENT_DIM])
# Z_A2 = mu_A2 + tf.multiply(noise_A2, tf.exp(.5*logstd_A2))
Y_5_A2 = tf.add(tf.matmul(test_noise_A2, DEC_W1_A2), DEC_B1_A2)
Y_6_A2 = tf.nn.relu(Y_5_A2)
Y_6_1_A2 = tf.nn.dropout(Y_6_A2, PKEEP)
Y_7_A2 = tf.add(tf.matmul(Y_6_1_A2, DEC_W2_A2), DEC_B2_A2)
Y_7_1_A2 = tf.multiply(Y_7_A2, ph_A2)
Y_rec_A2 = tf.nn.sigmoid(Y_7_1_A2)
# 0th Capsule KLT Loss
# KLT_A2 = -0.5*tf.reduce_sum(1 + 2*logstd_A2 - tf.pow(mu_A2, 2) - tf.exp(2*logstd_A2), axis = 1)
# 0th Capsule Attr Pred
# attr_loss_A2 = tf.nn.sigmoid_cross_entropy_with_logits(logits = attr_A2, labels = ph_A2)


# 0th Capsule Parameters
ph_A3 = tf.placeholder(tf.float32, shape = [None, 1], name = 'ph_A3')
test_noise_A3 = tf.placeholder(tf.float32, shape = [None, LATENT_DIM], name = 'test_noise_A3')
# ENC_W1_A3 = tf.get_variable(shape = [IMG_SIZE, ENC_DIM],  name = 'ENC_W1_A3')
# ENC_B1_A3 = tf.get_variable(shape = [ENC_DIM],  name = 'ENC_B1_A3')
# ENC_W2_A3 = tf.get_variable(shape = [ENC_DIM, LATENT_DIM],  name = 'ENC_W2_A3')
# ENC_B2_A3 = tf.get_variable(shape = [LATENT_DIM],  name = 'ENC_B2_A3')
# SAM_W_mu_A3 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_mu_A3')
# SAM_B_mu_A3 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_mu_A3')
# SAM_W_logstd_A3 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_logstd_A3')
# SAM_B_logstd_A3 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_logstd_A3')
DEC_W1_A3 = tf.get_variable(shape = [LATENT_DIM, ENC_DIM],  name = 'DEC_W1_A3')
DEC_B1_A3 = tf.get_variable(shape = [ENC_DIM],  name = 'DEC_B1_A3')
DEC_W2_A3 = tf.get_variable(shape = [ENC_DIM, IMG_SIZE],  name = 'DEC_W2_A3')
DEC_B2_A3 = tf.get_variable(shape = [IMG_SIZE],  name = 'DEC_B2_A3')
ATT_W1_A3 = tf.get_variable(shape = [ENC_DIM, 1],  name = 'ATT_W1_A3')
ATT_B1_A3 = tf.get_variable(shape = [1],  name = 'ATT_B1_A3')
# 0th Capsule Forward Pass
# Y_1_A3 = tf.add(tf.matmul(X, ENC_W1_A3), ENC_B1_A3)
# Y_2_A3 = tf.nn.relu(Y_1_A3)
# Y_2_1_A3 = tf.nn.dropout(Y_2_A3, PKEEP)
# Y_3_A3 = tf.add(tf.matmul(Y_2_1_A3, ENC_W2_A3), ENC_B2_A3)
# Y_4_A3 = tf.nn.tanh(Y_3_A3)
# attr_1_A3 = tf.add(tf.matmul(Y_2_1_A3, ATT_W1_A3), ATT_B1_A3)
# attr_A3 = tf.nn.sigmoid(attr_1_A3)
# Y_4_1_A3 = tf.nn.dropout(Y_4_A3, PKEEP)
# mu_A3 = tf.add(tf.matmul(Y_4_1_A3, SAM_W_mu_A3), SAM_B_mu_A3)
# logstd_A3 = tf.add(tf.matmul(Y_4_1_A3, SAM_W_logstd_A3), SAM_B_logstd_A3)
# noise_A3 = tf.random_normal([1, LATENT_DIM])
# Z_A3 = mu_A3 + tf.multiply(noise_A3, tf.exp(.5*logstd_A3))
Y_5_A3 = tf.add(tf.matmul(test_noise_A3, DEC_W1_A3), DEC_B1_A3)
Y_6_A3 = tf.nn.relu(Y_5_A3)
Y_6_1_A3 = tf.nn.dropout(Y_6_A3, PKEEP)
Y_7_A3 = tf.add(tf.matmul(Y_6_1_A3, DEC_W2_A3), DEC_B2_A3)
Y_7_1_A3 = tf.multiply(Y_7_A3, ph_A3)
Y_rec_A3 = tf.nn.sigmoid(Y_7_1_A3)
# 0th Capsule KLT Loss
# KLT_A3 = -0.5*tf.reduce_sum(1 + 2*logstd_A3 - tf.pow(mu_A3, 2) - tf.exp(2*logstd_A3), axis = 1)
# 0th Capsule Attr Pred
# attr_loss_A3 = tf.nn.sigmoid_cross_entropy_with_logits(logits = attr_A3, labels = ph_A3)


# 0th Capsule Parameters
ph_A4 = tf.placeholder(tf.float32, shape = [None, 1], name = 'ph_A4')
test_noise_A4 = tf.placeholder(tf.float32, shape = [None, LATENT_DIM], name = 'test_noise_A4')
# ENC_W1_A4 = tf.get_variable(shape = [IMG_SIZE, ENC_DIM],  name = 'ENC_W1_A4')
# ENC_B1_A4 = tf.get_variable(shape = [ENC_DIM],  name = 'ENC_B1_A4')
# ENC_W2_A4 = tf.get_variable(shape = [ENC_DIM, LATENT_DIM],  name = 'ENC_W2_A4')
# ENC_B2_A4 = tf.get_variable(shape = [LATENT_DIM],  name = 'ENC_B2_A4')
# SAM_W_mu_A4 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_mu_A4')
# SAM_B_mu_A4 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_mu_A4')
# SAM_W_logstd_A4 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_logstd_A4')
# SAM_B_logstd_A4 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_logstd_A4')
DEC_W1_A4 = tf.get_variable(shape = [LATENT_DIM, ENC_DIM],  name = 'DEC_W1_A4')
DEC_B1_A4 = tf.get_variable(shape = [ENC_DIM],  name = 'DEC_B1_A4')
DEC_W2_A4 = tf.get_variable(shape = [ENC_DIM, IMG_SIZE],  name = 'DEC_W2_A4')
DEC_B2_A4 = tf.get_variable(shape = [IMG_SIZE],  name = 'DEC_B2_A4')
ATT_W1_A4 = tf.get_variable(shape = [ENC_DIM, 1],  name = 'ATT_W1_A4')
ATT_B1_A4 = tf.get_variable(shape = [1],  name = 'ATT_B1_A4')
# 0th Capsule Forward Pass
# Y_1_A4 = tf.add(tf.matmul(X, ENC_W1_A4), ENC_B1_A4)
# Y_2_A4 = tf.nn.relu(Y_1_A4)
# Y_2_1_A4 = tf.nn.dropout(Y_2_A4, PKEEP)
# Y_3_A4 = tf.add(tf.matmul(Y_2_1_A4, ENC_W2_A4), ENC_B2_A4)
# Y_4_A4 = tf.nn.tanh(Y_3_A4)
# attr_1_A4 = tf.add(tf.matmul(Y_2_1_A4, ATT_W1_A4), ATT_B1_A4)
# attr_A4 = tf.nn.sigmoid(attr_1_A4)
# Y_4_1_A4 = tf.nn.dropout(Y_4_A4, PKEEP)
# mu_A4 = tf.add(tf.matmul(Y_4_1_A4, SAM_W_mu_A4), SAM_B_mu_A4)
# logstd_A4 = tf.add(tf.matmul(Y_4_1_A4, SAM_W_logstd_A4), SAM_B_logstd_A4)
# noise_A4 = tf.random_normal([1, LATENT_DIM])
# Z_A4 = mu_A4 + tf.multiply(noise_A4, tf.exp(.5*logstd_A4))
Y_5_A4 = tf.add(tf.matmul(test_noise_A4, DEC_W1_A4), DEC_B1_A4)
Y_6_A4 = tf.nn.relu(Y_5_A4)
Y_6_1_A4 = tf.nn.dropout(Y_6_A4, PKEEP)
Y_7_A4 = tf.add(tf.matmul(Y_6_1_A4, DEC_W2_A4), DEC_B2_A4)
Y_7_1_A4 = tf.multiply(Y_7_A4, ph_A4)
Y_rec_A4 = tf.nn.sigmoid(Y_7_1_A4)
# 0th Capsule KLT Loss
# KLT_A4 = -0.5*tf.reduce_sum(1 + 2*logstd_A4 - tf.pow(mu_A4, 2) - tf.exp(2*logstd_A4), axis = 1)
# 0th Capsule Attr Pred
# attr_loss_A4 = tf.nn.sigmoid_cross_entropy_with_logits(logits = attr_A4, labels = ph_A4)


# 0th Capsule Parameters
ph_A5 = tf.placeholder(tf.float32, shape = [None, 1], name = 'ph_A5')
test_noise_A5 = tf.placeholder(tf.float32, shape = [None, LATENT_DIM], name = 'test_noise_A5')
# ENC_W1_A5 = tf.get_variable(shape = [IMG_SIZE, ENC_DIM],  name = 'ENC_W1_A5')
# ENC_B1_A5 = tf.get_variable(shape = [ENC_DIM],  name = 'ENC_B1_A5')
# ENC_W2_A5 = tf.get_variable(shape = [ENC_DIM, LATENT_DIM],  name = 'ENC_W2_A5')
# ENC_B2_A5 = tf.get_variable(shape = [LATENT_DIM],  name = 'ENC_B2_A5')
# SAM_W_mu_A5 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_mu_A5')
# SAM_B_mu_A5 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_mu_A5')
# SAM_W_logstd_A5 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_logstd_A5')
# SAM_B_logstd_A5 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_logstd_A5')
DEC_W1_A5 = tf.get_variable(shape = [LATENT_DIM, ENC_DIM],  name = 'DEC_W1_A5')
DEC_B1_A5 = tf.get_variable(shape = [ENC_DIM],  name = 'DEC_B1_A5')
DEC_W2_A5 = tf.get_variable(shape = [ENC_DIM, IMG_SIZE],  name = 'DEC_W2_A5')
DEC_B2_A5 = tf.get_variable(shape = [IMG_SIZE],  name = 'DEC_B2_A5')
ATT_W1_A5 = tf.get_variable(shape = [ENC_DIM, 1],  name = 'ATT_W1_A5')
ATT_B1_A5 = tf.get_variable(shape = [1],  name = 'ATT_B1_A5')
# 0th Capsule Forward Pass
# Y_1_A5 = tf.add(tf.matmul(X, ENC_W1_A5), ENC_B1_A5)
# Y_2_A5 = tf.nn.relu(Y_1_A5)
# Y_2_1_A5 = tf.nn.dropout(Y_2_A5, PKEEP)
# Y_3_A5 = tf.add(tf.matmul(Y_2_1_A5, ENC_W2_A5), ENC_B2_A5)
# Y_4_A5 = tf.nn.tanh(Y_3_A5)
# attr_1_A5 = tf.add(tf.matmul(Y_2_1_A5, ATT_W1_A5), ATT_B1_A5)
# attr_A5 = tf.nn.sigmoid(attr_1_A5)
# Y_4_1_A5 = tf.nn.dropout(Y_4_A5, PKEEP)
# mu_A5 = tf.add(tf.matmul(Y_4_1_A5, SAM_W_mu_A5), SAM_B_mu_A5)
# logstd_A5 = tf.add(tf.matmul(Y_4_1_A5, SAM_W_logstd_A5), SAM_B_logstd_A5)
# noise_A5 = tf.random_normal([1, LATENT_DIM])
# Z_A5 = mu_A5 + tf.multiply(noise_A5, tf.exp(.5*logstd_A5))
Y_5_A5 = tf.add(tf.matmul(test_noise_A5, DEC_W1_A5), DEC_B1_A5)
Y_6_A5 = tf.nn.relu(Y_5_A5)
Y_6_1_A5 = tf.nn.dropout(Y_6_A5, PKEEP)
Y_7_A5 = tf.add(tf.matmul(Y_6_1_A5, DEC_W2_A5), DEC_B2_A5)
Y_7_1_A5 = tf.multiply(Y_7_A5, ph_A5)
Y_rec_A5 = tf.nn.sigmoid(Y_7_1_A5)
# 0th Capsule KLT Loss
# KLT_A5 = -0.5*tf.reduce_sum(1 + 2*logstd_A5 - tf.pow(mu_A5, 2) - tf.exp(2*logstd_A5), axis = 1)
# 0th Capsule Attr Pred
# attr_loss_A5 = tf.nn.sigmoid_cross_entropy_with_logits(logits = attr_A5, labels = ph_A5)


# 0th Capsule Parameters
ph_A6 = tf.placeholder(tf.float32, shape = [None, 1], name = 'ph_A6')
test_noise_A6 = tf.placeholder(tf.float32, shape = [None, LATENT_DIM], name = 'test_noise_A6')
# ENC_W1_A6 = tf.get_variable(shape = [IMG_SIZE, ENC_DIM],  name = 'ENC_W1_A6')
# ENC_B1_A6 = tf.get_variable(shape = [ENC_DIM],  name = 'ENC_B1_A6')
# ENC_W2_A6 = tf.get_variable(shape = [ENC_DIM, LATENT_DIM],  name = 'ENC_W2_A6')
# ENC_B2_A6 = tf.get_variable(shape = [LATENT_DIM],  name = 'ENC_B2_A6')
# SAM_W_mu_A6 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_mu_A6')
# SAM_B_mu_A6 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_mu_A6')
# SAM_W_logstd_A6 = tf.get_variable(shape = [LATENT_DIM, LATENT_DIM],  name = 'SAM_W_logstd_A6')
# SAM_B_logstd_A6 = tf.get_variable(shape = [LATENT_DIM],  name = 'SAM_B_logstd_A6')
DEC_W1_A6 = tf.get_variable(shape = [LATENT_DIM, ENC_DIM],  name = 'DEC_W1_A6')
DEC_B1_A6 = tf.get_variable(shape = [ENC_DIM],  name = 'DEC_B1_A6')
DEC_W2_A6 = tf.get_variable(shape = [ENC_DIM, IMG_SIZE],  name = 'DEC_W2_A6')
DEC_B2_A6 = tf.get_variable(shape = [IMG_SIZE],  name = 'DEC_B2_A6')
ATT_W1_A6 = tf.get_variable(shape = [ENC_DIM, 1],  name = 'ATT_W1_A6')
ATT_B1_A6 = tf.get_variable(shape = [1],  name = 'ATT_B1_A6')
# 0th Capsule Forward Pass
# Y_1_A6 = tf.add(tf.matmul(X, ENC_W1_A6), ENC_B1_A6)
# Y_2_A6 = tf.nn.relu(Y_1_A6)
# Y_2_1_A6 = tf.nn.dropout(Y_2_A6, PKEEP)
# Y_3_A6 = tf.add(tf.matmul(Y_2_1_A6, ENC_W2_A6), ENC_B2_A6)
# Y_4_A6 = tf.nn.tanh(Y_3_A6)
# attr_1_A6 = tf.add(tf.matmul(Y_2_1_A6, ATT_W1_A6), ATT_B1_A6)
# attr_A6 = tf.nn.sigmoid(attr_1_A6)
# Y_4_1_A6 = tf.nn.dropout(Y_4_A6, PKEEP)
# mu_A6 = tf.add(tf.matmul(Y_4_1_A6, SAM_W_mu_A6), SAM_B_mu_A6)
# logstd_A6 = tf.add(tf.matmul(Y_4_1_A6, SAM_W_logstd_A6), SAM_B_logstd_A6)
# noise_A6 = tf.random_normal([1, LATENT_DIM])
# Z_A6 = mu_A6 + tf.multiply(noise_A6, tf.exp(.5*logstd_A6))
Y_5_A6 = tf.add(tf.matmul(test_noise_A6, DEC_W1_A6), DEC_B1_A6)
Y_6_A6 = tf.nn.relu(Y_5_A6)
Y_6_1_A6 = tf.nn.dropout(Y_6_A6, PKEEP)
Y_7_A6 = tf.add(tf.matmul(Y_6_1_A6, DEC_W2_A6), DEC_B2_A6)
Y_7_1_A6 = tf.multiply(Y_7_A6, ph_A6)
Y_rec_A6 = tf.nn.sigmoid(Y_7_1_A6)
# 0th Capsule KLT Loss
# KLT_A6 = -0.5*tf.reduce_sum(1 + 2*logstd_A6 - tf.pow(mu_A6, 2) - tf.exp(2*logstd_A6), axis = 1)
# 0th Capsule Attr Pred
# attr_loss_A6 = tf.nn.sigmoid_cross_entropy_with_logits(logits = attr_A6, labels = ph_A6)


# Generate the image
Y_rec = tf.nn.sigmoid(Y_7_1_A0 + Y_7_1_A1 + Y_7_1_A2 + Y_7_1_A3 + Y_7_1_A4 + Y_7_1_A5 + Y_7_1_A6)


# Define session and saver
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "Models/Iteration_10000/Intermediate_Model_10000")


l = 100


# Look what each attribute generates!
# A0
attr_0 = np.ones([l, 1]).astype(np.float32)
attr_1 = np.zeros([l, 1]).astype(np.float32)
attr_2 = np.zeros([l, 1]).astype(np.float32)
attr_3 = np.zeros([l, 1]).astype(np.float32)
attr_4 = np.zeros([l, 1]).astype(np.float32)
attr_5 = np.zeros([l, 1]).astype(np.float32)
attr_6 = np.zeros([l, 1]).astype(np.float32)
a_noise = np.random.normal(0, 1, [l, LATENT_DIM])
generated_images_A0 = sess.run(Y_rec, feed_dict = {ph_A0 : attr_0 , ph_A1 : attr_1 , ph_A2 : attr_2 , ph_A3 : attr_3 , ph_A4 : attr_4 , ph_A5 : attr_5 , ph_A6 : attr_6 , test_noise_A0 : a_noise , test_noise_A1 : a_noise , test_noise_A2 : a_noise , test_noise_A3 : a_noise , test_noise_A4 : a_noise , test_noise_A5 : a_noise , test_noise_A6 : a_noise})
# print('[DEBUG] Shape of generated dataset : ' + str(generated_images_A0.shape)) # It is a list, not array
# print('[DEBUG] Exiting ... ')
# sys.exit()
index = 0
for i in range(l):
	im = generated_images_A0[index]
	index = index + 1
	cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im, [28, 28, 1])*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_A0.mp4')
time.sleep(0.5)
os.system('rm -f temp*.jpg')
time.sleep(0.5)
# A1
attr_0 = np.zeros([l, 1]).astype(np.float32)
attr_1 = np.ones([l, 1]).astype(np.float32)
attr_2 = np.zeros([l, 1]).astype(np.float32)
attr_3 = np.zeros([l, 1]).astype(np.float32)
attr_4 = np.zeros([l, 1]).astype(np.float32)
attr_5 = np.zeros([l, 1]).astype(np.float32)
attr_6 = np.zeros([l, 1]).astype(np.float32)
a_noise = np.random.normal(0, 1, [l, LATENT_DIM])
generated_images_A1 = sess.run(Y_rec, feed_dict = {ph_A0 : attr_0 , ph_A1 : attr_1 , ph_A2 : attr_2 , ph_A3 : attr_3 , ph_A4 : attr_4 , ph_A5 : attr_5 , ph_A6 : attr_6 , test_noise_A0 : a_noise , test_noise_A1 : a_noise , test_noise_A2 : a_noise , test_noise_A3 : a_noise , test_noise_A4 : a_noise , test_noise_A5 : a_noise , test_noise_A6 : a_noise})
index = 0
for i in range(l):
	im = generated_images_A1[index]
	index = index + 1
	cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im, [28, 28, 1])*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_A1.mp4')
time.sleep(0.5)
os.system('rm -f temp*.jpg')
time.sleep(0.5)
# A2
attr_0 = np.zeros([l, 1]).astype(np.float32)
attr_1 = np.zeros([l, 1]).astype(np.float32)
attr_2 = np.ones([l, 1]).astype(np.float32)
attr_3 = np.zeros([l, 1]).astype(np.float32)
attr_4 = np.zeros([l, 1]).astype(np.float32)
attr_5 = np.zeros([l, 1]).astype(np.float32)
attr_6 = np.zeros([l, 1]).astype(np.float32)
a_noise = np.random.normal(0, 1, [l, LATENT_DIM])
generated_images_A2 = sess.run(Y_rec, feed_dict = {ph_A0 : attr_0 , ph_A1 : attr_1 , ph_A2 : attr_2 , ph_A3 : attr_3 , ph_A4 : attr_4 , ph_A5 : attr_5 , ph_A6 : attr_6 , test_noise_A0 : a_noise , test_noise_A1 : a_noise , test_noise_A2 : a_noise , test_noise_A3 : a_noise , test_noise_A4 : a_noise , test_noise_A5 : a_noise , test_noise_A6 : a_noise})
index = 0
for i in range(l):
	im = generated_images_A2[index]
	index = index + 1
	cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im, [28, 28, 1])*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_A2.mp4')
time.sleep(0.5)
os.system('rm -f temp*.jpg')
time.sleep(0.5)
# A3
attr_0 = np.zeros([l, 1]).astype(np.float32)
attr_1 = np.zeros([l, 1]).astype(np.float32)
attr_2 = np.zeros([l, 1]).astype(np.float32)
attr_3 = np.ones([l, 1]).astype(np.float32)
attr_4 = np.zeros([l, 1]).astype(np.float32)
attr_5 = np.zeros([l, 1]).astype(np.float32)
attr_6 = np.zeros([l, 1]).astype(np.float32)
a_noise = np.random.normal(0, 1, [l, LATENT_DIM])
generated_images_A3 = sess.run(Y_rec, feed_dict = {ph_A0 : attr_0 , ph_A1 : attr_1 , ph_A2 : attr_2 , ph_A3 : attr_3 , ph_A4 : attr_4 , ph_A5 : attr_5 , ph_A6 : attr_6 , test_noise_A0 : a_noise , test_noise_A1 : a_noise , test_noise_A2 : a_noise , test_noise_A3 : a_noise , test_noise_A4 : a_noise , test_noise_A5 : a_noise , test_noise_A6 : a_noise})
index = 0
for i in range(l):
	im = generated_images_A3[index]
	index = index + 1
	cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im, [28, 28, 1])*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_A3.mp4')
time.sleep(0.5)
os.system('rm -f temp*.jpg')
time.sleep(0.5)
# A4
attr_0 = np.zeros([l, 1]).astype(np.float32)
attr_1 = np.zeros([l, 1]).astype(np.float32)
attr_2 = np.zeros([l, 1]).astype(np.float32)
attr_3 = np.zeros([l, 1]).astype(np.float32)
attr_4 = np.ones([l, 1]).astype(np.float32)
attr_5 = np.zeros([l, 1]).astype(np.float32)
attr_6 = np.zeros([l, 1]).astype(np.float32)
a_noise = np.random.normal(0, 1, [l, LATENT_DIM])
generated_images_A4 = sess.run(Y_rec, feed_dict = {ph_A0 : attr_0 , ph_A1 : attr_1 , ph_A2 : attr_2 , ph_A3 : attr_3 , ph_A4 : attr_4 , ph_A5 : attr_5 , ph_A6 : attr_6 , test_noise_A0 : a_noise , test_noise_A1 : a_noise , test_noise_A2 : a_noise , test_noise_A3 : a_noise , test_noise_A4 : a_noise , test_noise_A5 : a_noise , test_noise_A6 : a_noise})
index = 0
for i in range(l):
	im = generated_images_A4[index]
	index = index + 1
	cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im, [28, 28, 1])*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_A4.mp4')
time.sleep(0.5)
os.system('rm -f temp*.jpg')
time.sleep(0.5)
# A5
attr_0 = np.zeros([l, 1]).astype(np.float32)
attr_1 = np.zeros([l, 1]).astype(np.float32)
attr_2 = np.zeros([l, 1]).astype(np.float32)
attr_3 = np.zeros([l, 1]).astype(np.float32)
attr_4 = np.zeros([l, 1]).astype(np.float32)
attr_5 = np.ones([l, 1]).astype(np.float32)
attr_6 = np.zeros([l, 1]).astype(np.float32)
a_noise = np.random.normal(0, 1, [l, LATENT_DIM])
generated_images_A5 = sess.run(Y_rec, feed_dict = {ph_A0 : attr_0 , ph_A1 : attr_1 , ph_A2 : attr_2 , ph_A3 : attr_3 , ph_A4 : attr_4 , ph_A5 : attr_5 , ph_A6 : attr_6 , test_noise_A0 : a_noise , test_noise_A1 : a_noise , test_noise_A2 : a_noise , test_noise_A3 : a_noise , test_noise_A4 : a_noise , test_noise_A5 : a_noise , test_noise_A6 : a_noise})
index = 0
for i in range(l):
	im = generated_images_A5[index]
	index = index + 1
	cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im, [28, 28, 1])*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_A5.mp4')
time.sleep(0.5)
os.system('rm -f temp*.jpg')
time.sleep(0.5)
# A6
attr_0 = np.zeros([l, 1]).astype(np.float32)
attr_1 = np.zeros([l, 1]).astype(np.float32)
attr_2 = np.zeros([l, 1]).astype(np.float32)
attr_3 = np.zeros([l, 1]).astype(np.float32)
attr_4 = np.zeros([l, 1]).astype(np.float32)
attr_5 = np.zeros([l, 1]).astype(np.float32)
attr_6 = np.ones([l, 1]).astype(np.float32)
a_noise = np.random.normal(0, 1, [l, LATENT_DIM])
generated_images_A6 = sess.run(Y_rec, feed_dict = {ph_A0 : attr_0 , ph_A1 : attr_1 , ph_A2 : attr_2 , ph_A3 : attr_3 , ph_A4 : attr_4 , ph_A5 : attr_5 , ph_A6 : attr_6 , test_noise_A0 : a_noise , test_noise_A1 : a_noise , test_noise_A2 : a_noise , test_noise_A3 : a_noise , test_noise_A4 : a_noise , test_noise_A5 : a_noise , test_noise_A6 : a_noise})
index = 0
for i in range(l):
	im = generated_images_A6[index]
	index = index + 1
	cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im, [28, 28, 1])*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_A6.mp4')
time.sleep(0.5)
os.system('rm -f temp*.jpg')
time.sleep(0.5)
# Digits
for this_batch_int in range(10):
	if this_batch_int == 0 :
		attr_0 = np.ones([l, 1]).astype(np.float32)
		attr_1 = np.ones([l, 1]).astype(np.float32)
		attr_2 = np.ones([l, 1]).astype(np.float32)
		attr_3 = np.zeros([l, 1]).astype(np.float32)
		attr_4 = np.ones([l, 1]).astype(np.float32)
		attr_5 = np.ones([l, 1]).astype(np.float32)
		attr_6 = np.ones([l, 1]).astype(np.float32)
	if this_batch_int == 1 :
		attr_0 = np.zeros([l, 1]).astype(np.float32)
		attr_1 = np.zeros([l, 1]).astype(np.float32)
		attr_2 = np.ones([l, 1]).astype(np.float32)
		attr_3 = np.zeros([l, 1]).astype(np.float32)
		attr_4 = np.zeros([l, 1]).astype(np.float32)
		attr_5 = np.ones([l, 1]).astype(np.float32)
		attr_6 = np.zeros([l, 1]).astype(np.float32)
	if this_batch_int == 2 :
		attr_0 = np.ones([l, 1]).astype(np.float32)
		attr_1 = np.zeros([l, 1]).astype(np.float32)
		attr_2 = np.ones([l, 1]).astype(np.float32)
		attr_3 = np.ones([l, 1]).astype(np.float32)
		attr_4 = np.ones([l, 1]).astype(np.float32)
		attr_5 = np.zeros([l, 1]).astype(np.float32)
		attr_6 = np.ones([l, 1]).astype(np.float32)
	if this_batch_int == 3 :
		attr_0 = np.ones([l, 1]).astype(np.float32)
		attr_1 = np.zeros([l, 1]).astype(np.float32)
		attr_2 = np.ones([l, 1]).astype(np.float32)
		attr_3 = np.ones([l, 1]).astype(np.float32)
		attr_4 = np.zeros([l, 1]).astype(np.float32)
		attr_5 = np.ones([l, 1]).astype(np.float32)
		attr_6 = np.ones([l, 1]).astype(np.float32)
	if this_batch_int == 4 :
		attr_0 = np.zeros([l, 1]).astype(np.float32)
		attr_1 = np.ones([l, 1]).astype(np.float32)
		attr_2 = np.ones([l, 1]).astype(np.float32)
		attr_3 = np.ones([l, 1]).astype(np.float32)
		attr_4 = np.zeros([l, 1]).astype(np.float32)
		attr_5 = np.ones([l, 1]).astype(np.float32)
		attr_6 = np.zeros([l, 1]).astype(np.float32)
	if this_batch_int == 5 :
		attr_0 = np.ones([l, 1]).astype(np.float32)
		attr_1 = np.ones([l, 1]).astype(np.float32)
		attr_2 = np.zeros([l, 1]).astype(np.float32)
		attr_3 = np.ones([l, 1]).astype(np.float32)
		attr_4 = np.zeros([l, 1]).astype(np.float32)
		attr_5 = np.ones([l, 1]).astype(np.float32)
		attr_6 = np.ones([l, 1]).astype(np.float32)
	if this_batch_int == 6 :
		attr_0 = np.ones([l, 1]).astype(np.float32)
		attr_1 = np.ones([l, 1]).astype(np.float32)
		attr_2 = np.zeros([l, 1]).astype(np.float32)
		attr_3 = np.ones([l, 1]).astype(np.float32)
		attr_4 = np.ones([l, 1]).astype(np.float32)
		attr_5 = np.ones([l, 1]).astype(np.float32)
		attr_6 = np.ones([l, 1]).astype(np.float32)
	if this_batch_int == 7 :
		attr_0 = np.ones([l, 1]).astype(np.float32)
		attr_1 = np.zeros([l, 1]).astype(np.float32)
		attr_2 = np.ones([l, 1]).astype(np.float32)
		attr_3 = np.zeros([l, 1]).astype(np.float32)
		attr_4 = np.zeros([l, 1]).astype(np.float32)
		attr_5 = np.ones([l, 1]).astype(np.float32)
		attr_6 = np.zeros([l, 1]).astype(np.float32)
	if this_batch_int == 8 :
		attr_0 = np.ones([l, 1]).astype(np.float32)
		attr_1 = np.ones([l, 1]).astype(np.float32)
		attr_2 = np.ones([l, 1]).astype(np.float32)
		attr_3 = np.ones([l, 1]).astype(np.float32)
		attr_4 = np.ones([l, 1]).astype(np.float32)
		attr_5 = np.ones([l, 1]).astype(np.float32)
		attr_6 = np.ones([l, 1]).astype(np.float32)
	if this_batch_int == 9 :
		attr_0 = np.ones([l, 1]).astype(np.float32)
		attr_1 = np.ones([l, 1]).astype(np.float32)
		attr_2 = np.ones([l, 1]).astype(np.float32)
		attr_3 = np.ones([l, 1]).astype(np.float32)
		attr_4 = np.zeros([l, 1]).astype(np.float32)
		attr_5 = np.ones([l, 1]).astype(np.float32)
		attr_6 = np.zeros([l, 1]).astype(np.float32)
	a_noise_A0 = np.random.normal(0, 1, [l, LATENT_DIM])
	a_noise_A1 = np.random.normal(0, 1, [l, LATENT_DIM])
	a_noise_A2 = np.random.normal(0, 1, [l, LATENT_DIM])
	a_noise_A3 = np.random.normal(0, 1, [l, LATENT_DIM])
	a_noise_A4 = np.random.normal(0, 1, [l, LATENT_DIM])
	a_noise_A5 = np.random.normal(0, 1, [l, LATENT_DIM])
	a_noise_A6 = np.random.normal(0, 1, [l, LATENT_DIM])
	generated_images_ = sess.run(Y_rec, feed_dict = {ph_A0 : attr_0 , ph_A1 : attr_1 , ph_A2 : attr_2 , ph_A3 : attr_3 , ph_A4 : attr_4 , ph_A5 : attr_5 , ph_A6 : attr_6 , test_noise_A0 : a_noise_A0 , test_noise_A1 : a_noise_A1 , test_noise_A2 : a_noise_A2 , test_noise_A3 : a_noise_A3 , test_noise_A4 : a_noise_A4 , test_noise_A5 : a_noise_A5 , test_noise_A6 : a_noise_A6})
	index = 0
	for i in range(l):
		im = generated_images_[index]
		index = index + 1
		cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im, [28, 28, 1])*255)
	os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_Digit_' +  str(this_batch_int) + '.mp4')
	time.sleep(0.5)
	os.system('rm -f temp*.jpg')
	time.sleep(0.5)
# All Zeros
attr_0 = np.zeros([l, 1]).astype(np.float32)
attr_1 = np.zeros([l, 1]).astype(np.float32)
attr_2 = np.zeros([l, 1]).astype(np.float32)
attr_3 = np.zeros([l, 1]).astype(np.float32)
attr_4 = np.zeros([l, 1]).astype(np.float32)
attr_5 = np.zeros([l, 1]).astype(np.float32)
attr_6 = np.zeros([l, 1]).astype(np.float32)
a_noise = np.random.normal(0, 1, [l, LATENT_DIM])
generated_images_all_zero = sess.run(Y_rec, feed_dict = {ph_A0 : attr_0 , ph_A1 : attr_1 , ph_A2 : attr_2 , ph_A3 : attr_3 , ph_A4 : attr_4 , ph_A5 : attr_5 , ph_A6 : attr_6 , test_noise_A0 : a_noise , test_noise_A1 : a_noise , test_noise_A2 : a_noise , test_noise_A3 : a_noise , test_noise_A4 : a_noise , test_noise_A5 : a_noise , test_noise_A6 : a_noise})
index = 0
for i in range(l):
	im = generated_images_all_zero[index]
	index = index + 1
	cv2.imwrite('temp' + str(index).zfill(5) + '.jpg', np.reshape(im, [28, 28, 1])*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_All_Zeros.mp4')
time.sleep(0.5)
os.system('rm -f temp*.jpg')
time.sleep(0.5)
####################################################################################################
####################################################################################################
####################################################################################################


# a_batch_int, a_batch, a_attr_A0, a_attr_A1, a_attr_A2, a_attr_A3, a_attr_A4, a_attr_A5, a_attr_A6 = data_loader.GetNextPermBatchOfRandomSize()


# # Define decoder forward pass
# Y_5 = tf.add(tf.matmul(Z, DEC_W1), DEC_B1)
# Y_6 = tf.nn.tanh(Y_5)
# Y_6_1 = tf.nn.dropout(Y_6, PKEEP)
# Y_7 = tf.add(tf.matmul(Y_6_1, DEC_W2), DEC_B2)
# Y_8 = tf.nn.relu(Y_7)
# Y_8_1 = tf.nn.dropout(Y_8, PKEEP)
# Y_REC_1 = tf.add(tf.matmul(Y_8_1, OUT_W1), OUT_B1)
# Y_REC = tf.nn.sigmoid(Y_REC_1)


# # Define individual inputs!!
# all_zero = np.reshape(np.array([1, 0, 0, 1, 0, 0, 1]), [1, 7])
# # one = np.reshape(np.array([0, 0, 1, 0, 0, 1, 0]), [1, 7])
# # two = np.reshape(np.array([1, 0, 1, 1, 1, 0, 1]), [1, 7])
# # three = np.reshape(np.array([1, 0, 1, 1, 0, 1, 1]), [1, 7])
# # four = np.reshape(np.array([0, 1, 1, 1, 0, 1, 0]), [1, 7])
# # five = np.reshape(np.array([1, 1, 0, 1, 0, 1, 1]), [1, 7])
# # six = np.reshape(np.array([1, 1, 0, 1, 1, 1, 1]), [1, 7])
# # seven = np.reshape(np.array([1, 0, 1, 0, 0, 1, 0]), [1, 7])
# # eight = np.reshape(np.array([1, 1, 1, 1, 1, 1, 1]), [1, 7])
# # nine = np.reshape(np.array([1, 1, 1, 1, 0, 1, 0]), [1, 7])
# all_zero_ = sess.run(Y_REC, feed_dict = { Z : all_zero })
# # one_ = sess.run(Y_REC, feed_dict = { Z : one })
# # two_ = sess.run(Y_REC, feed_dict = { Z : two })
# # three_ = sess.run(Y_REC, feed_dict = { Z : three })
# # four_ = sess.run(Y_REC, feed_dict = { Z : four })
# # five_ = sess.run(Y_REC, feed_dict = { Z : five })
# # six_ = sess.run(Y_REC, feed_dict = { Z : six })
# # seven_ = sess.run(Y_REC, feed_dict = { Z : seven })
# # eight_ = sess.run(Y_REC, feed_dict = { Z : eight })
# # nine_ = sess.run(Y_REC, feed_dict = { Z : nine })
# cv2.imwrite('temp00001.jpg', np.reshape(all_zero_, [28, 28, 1])*255)
# # cv2.imwrite('temp00002.jpg', np.reshape(one_, [28, 28, 1])*255)
# # cv2.imwrite('temp00003.jpg', np.reshape(two_, [28, 28, 1])*255)
# # cv2.imwrite('temp00004.jpg', np.reshape(three_, [28, 28, 1])*255)
# # cv2.imwrite('temp00005.jpg', np.reshape(four_, [28, 28, 1])*255)
# # cv2.imwrite('temp00006.jpg', np.reshape(five_, [28, 28, 1])*255)
# # cv2.imwrite('temp00007.jpg', np.reshape(six_, [28, 28, 1])*255)
# # cv2.imwrite('temp00008.jpg', np.reshape(seven_, [28, 28, 1])*255)
# # cv2.imwrite('temp00009.jpg', np.reshape(eight_, [28, 28, 1])*255)
# # cv2.imwrite('temp00010.jpg', np.reshape(nine_, [28, 28, 1])*255)
# os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_Digit_All_Zero.mp4')
# time.sleep(0.5)
# os.system('rm -f temp00001.jpg temp00002.jpg temp00003.jpg temp00004.jpg temp00005.jpg temp00006.jpg temp00007.jpg temp00008.jpg temp00009.jpg temp00010.jpg')
# time.sleep(0.5)
