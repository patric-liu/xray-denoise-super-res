import json 
import random
import sys
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import keras
from keras import layers
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Activation, LeakyReLU
from keras.models import Model, model_from_json

class Network(object):

	def __init__(self, enc_sizes, deconv_size, dec_sizes, activations='sigmoid', test=0):
		# encoder shape, [input - latent)
		self.enc_sizes = enc_sizes
		# latent params
		self.deconv_size = deconv_size
		# decoder shape, [latent - output]
		self.dec_sizes = dec_sizes
		# creates three models with the same functional layers
		self.network = None
		self.activations = activations
		if test == 1:	
			self.build_linear(enc_sizes[0],dec_sizes[-1])
		elif test == 2:
			self.build_conv(enc_sizes[0],dec_sizes[-1])
		elif test == 3:
			self.build_deconv()
		elif test == 4:
			self.build_resdeconv()
		elif test == 0:
			pass

	def build_resdeconv(self):
		# Input layer
		layers_64 = [Input(shape=self.enc_sizes[0])]
		# 64
		for (filters, kernel) in self.enc_sizes[1:]:
			layers_64.append(self.residual_layer(layers_64[-1],filters,kernel))
		# 
		layers_128 = [Conv2DTranspose(filters = self.deconv_size[0], \
			kernel_size = self.deconv_size[1], strides = 2, padding='same')(layers_64[-1])]
		
		for (filters, kernel) in self.dec_sizes[:-1]:
			layers_128.append(self.residual_layer(layers_128[-1],filters,kernel))
		
		layers_128.append(Conv2D(filters=self.dec_sizes[-1][0], \
			kernel_size=self.dec_sizes[-1][1], padding='same', activation='hard_sigmoid')(layers_128[-1]))
		
		self.network = Model(layers_64[0],layers_128[-1])

	def residual_layer(self, inpt, filters, kernel):
		shortcut = inpt
		l1 = Conv2D(filters=filters, kernel_size=kernel, padding = 'same')(inpt)
		l1 = LeakyReLU(alpha=0.1)(l1)
		l2 = Conv2D(filters=filters, kernel_size=kernel, padding = 'same')(l1)
		l2 = LeakyReLU(alpha=0.1)(l2)
		l3 = layers.add([l2,shortcut])
		return l3

	def build_conv(self, inpt, outpt):
		self.start = Input(shape=inpt)
		self.conv1 = Conv2D(filters = 4, kernel_size = (5,5), padding='same')(self.start)
		self.conv2 = Conv2D(filters = 8, kernel_size = (5,5), padding='same')(self.conv1)
		self.conv3 = Conv2D(filters = 16, kernel_size = (5,5),padding='same')(self.conv2)
		self.conv4 = Conv2D(filters = 1, kernel_size = (5,5), padding='same')(self.conv3)
		self.network = Model(self.start,self.conv4)

	def build_deconv(self, activation = 'relu'):
		# input layer
		layers_64 = [Input(shape=self.enc_sizes[0])]
		# 64x64 convolution
		for (filters, kernel) in self.enc_sizes[1:]:
			layers_64.append(Conv2D(filters=filters, kernel_size=kernel, \
				padding='same', activation=activation)(layers_64[-1]))
		# transpose convolution
		layers_128 = [Conv2DTranspose(filters = self.deconv_size[0], \
			kernel_size = self.deconv_size[1], strides = 2, padding='same', activation=activation)(layers_64[-1])]
		# 128x128 convolution
		for (filters, kernel) in self.dec_sizes[:-1]:
			layers_128.append(Conv2D(filters=filters, kernel_size=kernel, \
				padding='same', activation=activation)(layers_128[-1]))
		# Last sigmoid layer
		layers_128.append(Conv2D(filters=self.dec_sizes[-1][0], kernel_size=self.dec_sizes[M<-1][1], padding='same', activation=self.activations)(layers_128[-1]))
		self.network = Model(layers_64[0],layers_128[-1])

	def build_linear(self, inpt, outpt):
		self.start = Input(shape=(inpt,))
		self.matrix = Dense(outpt, activation='linear')
		self.end = self.matrix(self.start)
		self.network = Model(self.start,self.end)

	def train(self, x_train,y_train, callback, val_split = 0.1, epochs=10, batch_size=100, verbose=2, optimizer = 'adadelta', loss = 'mean_squared_error'):
		# x_train is inputs
		# x_test is outputs
		self.network.compile(optimizer=optimizer, loss=loss)
		self.network.fit(x_train, y_train,
						epochs = epochs,
						batch_size = batch_size,
						shuffle=True,
						validation_split = val_split,
						verbose = verbose,
						callbacks = callback
						)

	def predict(self, image):
		return self.network.predict(image)