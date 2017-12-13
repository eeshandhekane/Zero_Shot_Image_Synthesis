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


# Reset the graph
tf.reset_default_graph()


# Placeholders and variables
# X = tf.placeholder(tf.float32, shape = [None, IMG_SIZE], name = 'X')
# ENC_W1 = tf.get_variable(shape = [IMG_SIZE, HIDDEN_DIM], name = 'ENC_W1')
# ENC_B1 = tf.get_variable(shape = [HIDDEN_DIM], name = 'ENC_B1')
# ENC_W2 = tf.get_variable(shape = [HIDDEN_DIM, HIDDEN_DIM], name = 'ENC_W2')
# ENC_B2 = tf.get_variable(shape = [HIDDEN_DIM], name = 'ENC_B2')
# SAM_W_mu = tf.get_variable(shape = [HIDDEN_DIM, LATENT_DIM], name = 'SAM_W_mu')
# SAM_B_mu = tf.get_variable(shape = [LATENT_DIM], name = 'SAM_B_mu')
# SAM_W_logstd = tf.get_variable(shape = [HIDDEN_DIM, LATENT_DIM], name = 'SAM_W_logstd')
# SAM_B_logstd = tf.get_variable(shape = [LATENT_DIM], name = 'SAM_B_logstd')
DEC_W1 = tf.get_variable(shape = [LATENT_DIM, HIDDEN_DIM], name = 'DEC_W1')
DEC_B1 = tf.get_variable(shape = [HIDDEN_DIM], name = 'DEC_B1')
DEC_W2 = tf.get_variable(shape = [HIDDEN_DIM, HIDDEN_DIM], name = 'DEC_W2')
DEC_B2 = tf.get_variable(shape = [HIDDEN_DIM], name = 'DEC_B2')
OUT_W1 = tf.get_variable(shape = [HIDDEN_DIM, IMG_SIZE], name = 'OUT_W1')
OUT_B1 = tf.get_variable(shape = [IMG_SIZE], name = 'OUT_B1')
Z = tf.placeholder(tf.float32, shape = [None, LATENT_DIM])


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
