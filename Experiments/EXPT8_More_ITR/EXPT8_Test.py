# Dependencies
import tensorflow as tf
import numpy as np
import cv2
import os, sys, re
import time


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


# # Placeholders and variables
# # X = tf.placeholder(tf.float32, shape = [None, IMG_SIZE], name = 'X')
# # ENC_W1 = tf.get_variable(shape = [IMG_SIZE, HIDDEN_DIM], name = 'ENC_W1')
# # ENC_B1 = tf.get_variable(shape = [HIDDEN_DIM], name = 'ENC_B1')
# # ENC_W2 = tf.get_variable(shape = [HIDDEN_DIM, HIDDEN_DIM], name = 'ENC_W2')
# # ENC_B2 = tf.get_variable(shape = [HIDDEN_DIM], name = 'ENC_B2')
# # SAM_W_mu = tf.get_variable(shape = [HIDDEN_DIM, LATENT_DIM], name = 'SAM_W_mu')
# # SAM_B_mu = tf.get_variable(shape = [LATENT_DIM], name = 'SAM_B_mu')
# # SAM_W_logstd = tf.get_variable(shape = [HIDDEN_DIM, LATENT_DIM], name = 'SAM_W_logstd')
# # SAM_B_logstd = tf.get_variable(shape = [LATENT_DIM], name = 'SAM_B_logstd')
# DEC_W1 = tf.get_variable(shape = [LATENT_DIM, HIDDEN_DIM], name = 'DEC_W1')
# DEC_B1 = tf.get_variable(shape = [HIDDEN_DIM], name = 'DEC_B1')
# DEC_W2 = tf.get_variable(shape = [HIDDEN_DIM, HIDDEN_DIM], name = 'DEC_W2')
# DEC_B2 = tf.get_variable(shape = [HIDDEN_DIM], name = 'DEC_B2')
# OUT_W1 = tf.get_variable(shape = [HIDDEN_DIM, IMG_SIZE], name = 'OUT_W1')
# OUT_B1 = tf.get_variable(shape = [IMG_SIZE], name = 'OUT_B1')
# Z = tf.placeholder(tf.float32, shape = [None, LATENT_DIM])





# Define session and saver
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "Models/Iteration_95000/Intermediate_Model_95000")


# Define decoder forward pass
Y_5 = tf.add(tf.matmul(Z, DEC_W1), DEC_B1)
Y_6 = tf.nn.tanh(Y_5)
Y_6_1 = tf.nn.dropout(Y_6, PKEEP)
Y_7 = tf.add(tf.matmul(Y_6_1, DEC_W2), DEC_B2)
Y_8 = tf.nn.relu(Y_7)
Y_8_1 = tf.nn.dropout(Y_8, PKEEP)
Y_REC_1 = tf.add(tf.matmul(Y_8_1, OUT_W1), OUT_B1)
Y_REC = tf.nn.sigmoid(Y_REC_1)


# Define individual inputs!!
all_zero = np.reshape(np.array([1, 0, 0, 1, 0, 0, 1]), [1, 7])
# one = np.reshape(np.array([0, 0, 1, 0, 0, 1, 0]), [1, 7])
# two = np.reshape(np.array([1, 0, 1, 1, 1, 0, 1]), [1, 7])
# three = np.reshape(np.array([1, 0, 1, 1, 0, 1, 1]), [1, 7])
# four = np.reshape(np.array([0, 1, 1, 1, 0, 1, 0]), [1, 7])
# five = np.reshape(np.array([1, 1, 0, 1, 0, 1, 1]), [1, 7])
# six = np.reshape(np.array([1, 1, 0, 1, 1, 1, 1]), [1, 7])
# seven = np.reshape(np.array([1, 0, 1, 0, 0, 1, 0]), [1, 7])
# eight = np.reshape(np.array([1, 1, 1, 1, 1, 1, 1]), [1, 7])
# nine = np.reshape(np.array([1, 1, 1, 1, 0, 1, 0]), [1, 7])
all_zero_ = sess.run(Y_REC, feed_dict = { Z : all_zero })
# one_ = sess.run(Y_REC, feed_dict = { Z : one })
# two_ = sess.run(Y_REC, feed_dict = { Z : two })
# three_ = sess.run(Y_REC, feed_dict = { Z : three })
# four_ = sess.run(Y_REC, feed_dict = { Z : four })
# five_ = sess.run(Y_REC, feed_dict = { Z : five })
# six_ = sess.run(Y_REC, feed_dict = { Z : six })
# seven_ = sess.run(Y_REC, feed_dict = { Z : seven })
# eight_ = sess.run(Y_REC, feed_dict = { Z : eight })
# nine_ = sess.run(Y_REC, feed_dict = { Z : nine })
cv2.imwrite('temp00001.jpg', np.reshape(all_zero_, [28, 28, 1])*255)
# cv2.imwrite('temp00002.jpg', np.reshape(one_, [28, 28, 1])*255)
# cv2.imwrite('temp00003.jpg', np.reshape(two_, [28, 28, 1])*255)
# cv2.imwrite('temp00004.jpg', np.reshape(three_, [28, 28, 1])*255)
# cv2.imwrite('temp00005.jpg', np.reshape(four_, [28, 28, 1])*255)
# cv2.imwrite('temp00006.jpg', np.reshape(five_, [28, 28, 1])*255)
# cv2.imwrite('temp00007.jpg', np.reshape(six_, [28, 28, 1])*255)
# cv2.imwrite('temp00008.jpg', np.reshape(seven_, [28, 28, 1])*255)
# cv2.imwrite('temp00009.jpg', np.reshape(eight_, [28, 28, 1])*255)
# cv2.imwrite('temp00010.jpg', np.reshape(nine_, [28, 28, 1])*255)
os.system('ffmpeg -f image2 -r 1/2 -i temp%05d.jpg -vcodec mpeg4 -y ' + 'Test_Digit_All_Zero.mp4')
time.sleep(0.5)
os.system('rm -f temp00001.jpg temp00002.jpg temp00003.jpg temp00004.jpg temp00005.jpg temp00006.jpg temp00007.jpg temp00008.jpg temp00009.jpg temp00010.jpg')
time.sleep(0.5)
