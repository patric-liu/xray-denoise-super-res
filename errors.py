import getdata
import numpy as np 
import network
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2
import os
from keras.models import model_from_json, Model
from keras.losses import mean_squared_error as loss
from keras import backend as K
import pickle

# load data
flatten = False      # Get 1D data
rescale = True       #  Rescale 'RGB' values from [0,255] to [1,0]
denoise_only = False # Reshape target values from 128x128 to 64x64
amount = 0        # Amount of data to load, 0 gives loads all data
train_x, train_y = getdata.get_training(flatten = flatten, rescale = rescale, amount = amount, denoise_only = denoise_only)
test_data, rng = getdata.get_test(flatten = flatten, rescale = rescale, denoise_only = denoise_only, amount = amount if amount < 4000 else 3999) 
print('loaded data')

# Load Model
model_name = 'hugenet3'
# model reconstruction from JSON
net = network.Network(None, None, None, test = 0)
with open(os.getcwd() + '/model_weights/'+ model_name +'_architecture.json', 'r') as f:
    net.network = model_from_json(f.read())
# load weights into the model
net.network.load_weights(os.getcwd() + '/model_weights/'+ model_name +'_weights.h5')
net.network.summary()

# Compute Errors
y_true = K.variable(train_y)
predictions = net.predict(train_x)
y_pred = K.variable(predictions)
error = np.expand_dims(K.eval(loss(y_true, y_pred)), axis=3)
mean_error = np.squeeze(np.mean(error,(1,2)))
print('errors computed')

# GET ERROR INDICES
percentages = [0.25,0.1,0.05,0.01,0.001,0.0001]
ind = []
amount = amount if amount != 0 else 16000
for percent in percentages:
	m = int(percent * amount)
	num = m if m != 0 else 1
	ind.append(np.argpartition(mean_error, -num)[-num:])

for i in ind:
	print(np.shape(i))

with open('error_indices.pkl', 'wb') as output:
    pickle.dump(ind, output, pickle.HIGHEST_PROTOCOL)

'''del ind

with open('error_indices.pkl', 'rb') as input:
	ind = pickle.load(input)

for i in ind:
	print(np.shape(i))'''
