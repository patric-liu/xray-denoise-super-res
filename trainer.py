import getdata
import pickle
import numpy as np 
import network
import matplotlib.pyplot as plt
import scipy.ndimage
import cv2
import os
from keras.models import model_from_json, Model
from keras.callbacks import TensorBoard


############ LOAD UP PREVIOUS MODEL #########################
# Load Model
model_name = 'hugenet2'

# model reconstruction from JSON
net = network.Network(None, None, None, test = 0)
with open(os.getcwd() + '/model_weights/'+ model_name +'_architecture.json', 'r') as f:
    net.network = model_from_json(f.read())
# load weights into the model
net.network.load_weights(os.getcwd() + '/model_weights/'+ model_name +'_weights.h5')
net.network.summary()
#############################################################
'''

############ CREATE AND TRAIN NEW MODEL #####################
# define model
filters_64  = [(64,64,1),(32,5),(32,5),(32,5),(32,5),(32,5),(32,5)] # [(INPUT_SHAPE),(num_filters, kernel_size)...]
deconv      = (32,5) # (num_filters, kernel_size)
filters_128 = [(32,5),(32,5),(32,5),(32,5),(32,5),(32,5),(1,5)] #[(num_filters, kernel_size)...]

net = network.Network(filters_64, deconv, filters_128, test = 4)
net.network.summary()
#############################################################
'''

# OUTPUT MODEL NAME
model_name = 'hugenet3'

# load data
flatten = False      # Get 1D data
rescale = True       #  Rescale 'RGB' values from [0,255] to [1,0]
denoise_only = False # Reshape target values from 128x128 to 64x64
amount = 0        # Amount of data to load, 0 gives loads all data
train_x, train_y = getdata.get_training(flatten = flatten, rescale = rescale, \
    amount = amount, denoise_only = denoise_only)
test_data, rng = getdata.get_test(flatten = flatten, rescale = rescale, \
    denoise_only = denoise_only, amount = amount if amount < 4000 else 3999) 

# DEFINE CALLBACKS
callback = [TensorBoard(log_dir='./logs/run5',write_graph=False)]

# TRAIN THE AUTOENCODER ALTERNATINGLY
losses = {True: 'mean_squared_error', False: 'mean_absolute_error'}
loss_ = False
for _ in range(0):
    net.train(train_x,train_y, epochs = 4, verbose = 1,\
        loss = losses[loss_], optimizer= 'adadelta', batch_size = 16,\
        callback = callback)
    loss_ = not loss_

# TRAIN THE AUTOENCODER NORMALLY
for _ in range(8):
    net.train(train_x,train_y, epochs = 1, verbose = 1,\
            loss = 'mean_squared_error', optimizer= 'adadelta',\
            batch_size = 16, callback = callback)
    save(model_name)


# SAVE MODEL
def save(name):
    # save weights
    save_path = os.getcwd() + '/model_weights/'+ name +'_weights.h5'
    net.network.save_weights(save_path)
    # save architecture in JSON
    with open(os.getcwd() + '/model_weights/'+ name +'_architecture.json', 'w') as f:
        f.write(net.network.to_json())



