import json 
import random
import sys
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D
from keras.models import Model

class Autoencoder(object):

	def __init__(self, enc_sizes, latent_size, dec_sizes, test=0):
		# encoder shape, [input - latent)
		self.enc_sizes = enc_sizes
		# decoder shape, [latent - output]
		self.dec_sizes = dec_sizes
		# creates three models with the same functional layers
		if test == 0 :
			self.build_autoencoder(enc_sizes, dec_sizes, latent_size)
			
		elif test == 1:	
			self.build_test(enc_sizes[0],dec_sizes[-1])
		elif test == 2:
			self.build_conv(enc_sizes[0],dec_sizes[-1])

	def build_conv(self, inpt, outpt):
		self.start = Input(shape=inpt)
		self.conv1 = Conv2D(filters = 4, kernel_size = (5,5), padding='same')(self.start)
		self.conv2 = Conv2D(filters = 8, kernel_size = (5,5), padding='same')(self.conv1)
		self.conv3 = Conv2D(filters = 16, kernel_size = (5,5), padding='same')(self.conv2)
		self.conv4 = Conv2D(filters = 1, kernel_size = (5,5), padding='same')(self.conv3)
		self.autoencoder = Model(self.start,self.conv4)


	def build_autoencoder(self, enc_sizes, dec_sizes, latent_size):
		# encoder_layers store the [input - latent) : [input, h1, h2.. hn]
		# decoder_layers store the (latent - output]  : [hn+1, hn+2.. hn+m, output]

		# create encoder input layer and store it
		encoder_layers = [Input(shape=(enc_sizes[0],))]
		# creates encoder hidden layers and links them
		for n, size in enumerate(enc_sizes[1:]):
			encoder_layers.append(Dense(size, activation='linear')(encoder_layers[n]))
		
		# create latent layer and link to encoder and store it
		latent_layer = Dense(latent_size, activation='linear')(encoder_layers[-1])

		# creates decoder hidden layers and does not link them
		decoder_layers_ = []
		for size in dec_sizes[:-1]:
			decoder_layers_.append(Dense(size, activation='linear'))
		decoder_layers_.append(Dense(dec_sizes[-1], activation='linear'))

		# links decoder hidden layers to encoder
		decoder_layers_enc = [decoder_layers_[0](latent_layer)]
		for index, layer in enumerate(decoder_layers_[1:]):
			decoder_layers_enc.append([layer][0](decoder_layers_enc[index]))
		
		# build autoencoder and encoder models
		self.autoencoder = Model(encoder_layers[0], decoder_layers_enc[-1])
		self.encoder = Model(encoder_layers[0], latent_layer)

		# rebuild decoder models
		latent_layer_placeholder = Input(shape=(latent_size,))
		# links decoder hidden layers to latent layer
		decoder_layers_lat = [decoder_layers_[0](latent_layer_placeholder)]
		for index, layer in enumerate(decoder_layers_[1:]):
			decoder_layers_lat.append([layer][0](decoder_layers_lat[index]))
		self.decoder = Model(latent_layer_placeholder,decoder_layers_lat[-1])


	def build_test(self, inpt, outpt):
		self.start = Input(shape=(inpt,))
		self.matrix = Dense(outpt, activation='linear')
		self.end = self.matrix(self.start)
		self.autoencoder = Model(self.start,self.end)


	def train(self, x_train,y_train,epochs=10, batch_size=100, verbose=2, optimizer = 'adadelta', loss = 'mean_squared_error'):
		# x_train is inputs
		# x_test is outputs
		self.autoencoder.compile(optimizer=optimizer, loss=loss)
		self.autoencoder.fit(x_train, y_train,
						epochs = epochs,
						batch_size = batch_size,
						shuffle=True,
						verbose = verbose,
						)

	def predict(self, input_img):
		return self.autoencoder.predict(input_img)