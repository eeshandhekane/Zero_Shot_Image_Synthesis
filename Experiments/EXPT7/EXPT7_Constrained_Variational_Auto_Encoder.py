# Dependencies
import tensorflow as tf
import numpy as np
import cv2
import os, sys, re


# Load MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)


# Parameters
IMG_SIZE = 28*28 # MNIST Image Size, Linearized
LATENT_DIM = 7
HIDDEN_DIM = 500
ITR = 100000 + 1
RECORD_ITR = 5000
DISPLAY_ITR = 10
PKEEP = 0.7
LR = 0.001
VAE_LR = 0.001
attribute_LR = 0.001
BATCH_SIZE = 64


# Data loader class
class dataLoader(object) :
	"""
	The class to load numbers and get the attribute representations as well!!
	"""
	def __init__(self) :
		self.mnist = input_data.read_data_sets('/tmp/data', one_hot = True)


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


# Instantiate a data loader
data_loader = dataLoader()
trial_batch_X_, trial_batch_Y_, trial_batch_Y__, trial_batch_attribute_np = data_loader.GetNextBatch(5)
print trial_batch_Y__, '\n', trial_batch_Y_, '\n', trial_batch_attribute_np, '\n'
#sys.exit()


# Define placeholders and variables
X = tf.placeholder(tf.float32, shape = [None, IMG_SIZE], name = 'X')
ENC_W1 = tf.Variable(tf.truncated_normal([IMG_SIZE, HIDDEN_DIM], stddev = 0.1), name = 'ENC_W1')
ENC_B1 = tf.Variable(tf.truncated_normal([HIDDEN_DIM], stddev = 0.1), name = 'ENC_B1')
ENC_W2 = tf.Variable(tf.truncated_normal([HIDDEN_DIM, HIDDEN_DIM], stddev = 0.1), name = 'ENC_W2')
ENC_B2 = tf.Variable(tf.truncated_normal([HIDDEN_DIM], stddev = 0.1), name = 'ENC_B2')
SAM_W_mu = tf.Variable(tf.truncated_normal([HIDDEN_DIM, LATENT_DIM], stddev = 0.1), name = 'SAM_W_mu')
SAM_B_mu = tf.Variable(tf.truncated_normal([LATENT_DIM], stddev = 0.1), 'SAM_B_mu')
SAM_W_logstd = tf.Variable(tf.truncated_normal([HIDDEN_DIM, LATENT_DIM], stddev = 0.1), name = 'SAM_W_logstd')
SAM_B_logstd = tf.Variable(tf.truncated_normal([LATENT_DIM], stddev = 0.1), 'SAM_B_logstd')
DEC_W1 = tf.Variable(tf.truncated_normal([LATENT_DIM, HIDDEN_DIM], stddev = 0.1), name = 'DEC_W1')
DEC_B1 = tf.Variable(tf.truncated_normal([HIDDEN_DIM], stddev = 0.1), name = 'DEC_B1')
DEC_W2 = tf.Variable(tf.truncated_normal([HIDDEN_DIM, HIDDEN_DIM], stddev = 0.1), name = 'DEC_W2')
DEC_B2 = tf.Variable(tf.truncated_normal([HIDDEN_DIM], stddev = 0.1), name = 'DEC_B2')
OUT_W1 = tf.Variable(tf.truncated_normal([HIDDEN_DIM, IMG_SIZE], stddev = 0.1), name = 'OUT_W1')
OUT_B1 = tf.Variable(tf.truncated_normal([IMG_SIZE]), name = 'OUT_B1')
Y_attr_1_true = tf.placeholder(tf.float32, shape = [None, 1], name = 'Y_attr_1_true')
Y_attr_2_true = tf.placeholder(tf.float32, shape = [None, 1], name = 'Y_attr_2_true')
Y_attr_3_true = tf.placeholder(tf.float32, shape = [None, 1], name = 'Y_attr_3_true')
Y_attr_4_true = tf.placeholder(tf.float32, shape = [None, 1], name = 'Y_attr_4_true')
Y_attr_5_true = tf.placeholder(tf.float32, shape = [None, 1], name = 'Y_attr_5_true')
Y_attr_6_true = tf.placeholder(tf.float32, shape = [None, 1], name = 'Y_attr_6_true')
Y_attr_7_true = tf.placeholder(tf.float32, shape = [None, 1], name = 'Y_attr_7_true')
Y_true = tf.placeholder(tf.float32, [None, LATENT_DIM])


# Define the forward pass
Y_1 = tf.add(tf.matmul(X, ENC_W1), ENC_B1)
Y_2 = tf.nn.relu(Y_1)
Y_2_1 = tf.nn.dropout(Y_2, PKEEP)
Y_3 = tf.add(tf.matmul(Y_2_1, ENC_W2), ENC_B2)
Y_4 = tf.nn.tanh(Y_3)
Y_4_1 = tf.nn.dropout(Y_4, PKEEP)
mu = tf.add(tf.matmul(Y_4_1, SAM_W_mu), SAM_B_mu)
logstd = tf.add(tf.matmul(Y_4_1, SAM_W_logstd), SAM_B_logstd)
noise = tf.random_normal([1, LATENT_DIM])
Z = mu + tf.multiply(noise, tf.exp(.5*logstd))
Y_5 = tf.add(tf.matmul(Z, DEC_W1), DEC_B1)
Y_6 = tf.nn.tanh(Y_5)
Y_6_1 = tf.nn.dropout(Y_6, PKEEP)
Y_7 = tf.add(tf.matmul(Y_6_1, DEC_W2), DEC_B2)
Y_8 = tf.nn.relu(Y_7)
Y_8_1 = tf.nn.dropout(Y_8, PKEEP)
Y_REC_1 = tf.add(tf.matmul(Y_8_1, OUT_W1), OUT_B1)
Y_REC = tf.nn.sigmoid(Y_REC_1)


# Define loss and optimizers
LLT = tf.reduce_sum(X*tf.log(Y_REC + 1e-8) + (1 - X)*tf.log(1 - Y_REC + 1e-8), axis = 1)
# KLT = -0.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu, 2) - tf.exp(2*logstd), axis = 1)
# VLB = tf.reduce_mean(LLT - KLT)
VLB = tf.reduce_mean(LLT)
Constraint_T_L2 = tf.nn.l2_loss(mu - Y_true)
VAE_training_step = tf.train.AdamOptimizer(VAE_LR).minimize(-VLB)
attribute_training_step = tf.train.AdamOptimizer(attribute_LR).minimize(Constraint_T_L2)


# Initialize!!
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()


# Training loops
for iteration in range(ITR) :
	X_batch, Y_batch, Y__batch, attribute_batch = data_loader.GetNextBatch(BATCH_SIZE)
	for j in range(51):
		X_batch_a, Y_batch_a, Y__batch_a, attribute_batch_a = data_loader.GetNextBatch(BATCH_SIZE)
		sess.run(attribute_training_step, feed_dict = { X : X_batch_a , Y_true : attribute_batch_a })
		if j%10 == 0 :
			feed_dict = { X : X_batch_a , Y_true : attribute_batch_a }
			attribute_L2_loss = sess.run(Constraint_T_L2, feed_dict = feed_dict)
			print('[TRAINING] Attribute identification loss : ' + str(attribute_L2_loss))
	sess.run(VAE_training_step, feed_dict = { X : X_batch })
	if iteration%DISPLAY_ITR == 0:
		print('[TRAINING] Iteration : ' + str(iteration))
	if iteration%RECORD_ITR == 0:
		X_batch_te, Y_batch_te = mnist.train.next_batch(5)
		[LLT_, VLB_, Y_REC_] = sess.run([LLT, VLB, Y_REC], feed_dict = { X : X_batch_te })
		im_data = []
		im_data.append(X_batch_te)
		im_data.append(Y_REC_)
		im_data_np = np.array(im_data)
		np.save('Reconstructed_Dataset/Reconstructed_Images_' + str(iteration), im_data_np)
		os.system('mkdir Models/Iteration_' + str(iteration))
		saver.save(sess, 'Models/Iteration_' + str(iteration) + '/Intermediate_Model_' + str(iteration))