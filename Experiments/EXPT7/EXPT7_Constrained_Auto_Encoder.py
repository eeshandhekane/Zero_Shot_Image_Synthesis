# Dependencies
import tensorflow as tf
import numpy as np
import cv2
import os, sys, re


# Import MNIST
from tensorflow.examples.tutorials.mnist import input_data


# Define parameters
BATCH_SIZE = 32


# Define customized data loader class
class dataLoader(object) :
	"""
	Has several functions that load customized datasets
	"""
	# Constructor
	def __init__(self) :
		self.mnist = input_data.read_data_sets('/tmp/data', one_hot = True)
		self.attribute_list = []
		self.attribute_list.append([1, 1, 1, 0, 1, 1, 1]) # 0
		self.attribute_list.append([0, 0, 1, 0, 0, 1, 0]) # 1
		self.attribute_list.append([1, 0, 1, 1, 1, 0, 1]) # 2
		self.attribute_list.append([1, 0, 1, 1, 0, 1, 1]) # 3
		self.attribute_list.append([0, 1, 1, 1, 0, 1, 0]) # 4
		self.attribute_list.append([1, 1, 0, 1, 0, 1, 1]) # 5
		self.attribute_list.append([1, 1, 0, 1, 1, 1, 1]) # 6
		self.attribute_list.append([1, 0, 1, 0, 0, 1, 0]) # 7
		self.attribute_list.append([1, 1, 1, 1, 1, 1, 1]) # 8
		self.attribute_list.append([1, 1, 1, 1, 0, 1, 0]) # 9


	# Function to load next training batch
	def GetNextTrainingBatch(self, batch_size = BATCH_SIZE) :
		batch_X, batch_Y = self.mnist.train.next_batch(batch_size)
		return batch_X, batch_Y


	# Function to load attributed batch
	def GetNextAttributeBatch(self, batch_size = BATCH_SIZE) :
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
		return batch_X_, batch_Y_, batch_Y__, batch_attribute_np # X, Y_int, Y_one_hot, Y_attr


	# Function to load attributes as 