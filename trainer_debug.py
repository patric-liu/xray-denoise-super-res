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
from keras import backend as K

'''
############ LOAD UP PREVIOUS MODEL #########################
# Load Model
model_name = 'hugenet3'

# model reconstruction from JSON
net = network.Network(None, None, None, test = 0)
with open(os.getcwd() + '/model_weights/'+ model_name +'_architecture.json', 'r') as f:
    net.network = model_from_json(f.read())
# load weights into the model
# net.network.load_weights(os.getcwd() + '/model_weights/'+ model_name +'_weights.h5')
net.network.summary()
#############################################################
'''

############ CREATE AND TRAIN NEW MODEL #####################
# define model
filters_64  = [(64,64,1),(32,3),(32,3),(32,3),(32,3),(32,3)] # [(INPUT_SHAPE),(num_filters, kernel_size)...]
deconv      = (32,3) # (num_filters, kernel_size)
filters_128 = [(32,3),(32,3),(32,3),(32,3),(32,3),(32,3),(1,3)] #[(num_filters, kernel_size)...]

net = network.Network(filters_64, deconv, filters_128, test = 4)
net.network.summary()
#############################################################



# OUTPUT MODEL NAME
model_name = 'hugenet3RMSE_test'
# SAVE MODEL FUNCTION
def save(name):
    # save weights
    save_path = os.getcwd() + '/model_weights/'+ name +'_weights.h5'
    net.network.save_weights(save_path)
    # save architecture in JSON
    with open(os.getcwd() + '/model_weights/'+ name +'_architecture.json', 'w') as f:
        f.write(net.network.to_json())

# load data
flatten = False      # Get 1D data
rescale = True       #  Rescale 'RGB' values from [0,255] to [1,0]
denoise_only = False # Reshape target values from 128x128 to 64x64
amount = 0        # Amount of data to load, 0 gives loads all data
train_x, train_y = getdata.get_training(flatten = flatten, rescale = rescale, \
    amount = amount, denoise_only = denoise_only)
test_data, rng = getdata.get_test(flatten = flatten, rescale = rescale, \
    denoise_only = denoise_only, amount = amount if amount < 4000 else 3999) 

val_x, val_y = train_x[14000:15999], train_y[14000:15999]
train_x, train_y = train_x[0:14000], train_y[0:14000]

# DEFINE CALLBACKS
callback = [TensorBoard(log_dir='./logs/run5',write_graph=False)]

def RMSError(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true)))

# TRAIN THE AUTOENCODER ALTERNATINGLY
losses = {True: RMSError, False: 'mean_squared_error'}
loss_ = False

net.network.compile(optimizer = 'adadelta', loss = RMSError)
print('RMSE', net.network.evaluate(val_x, val_y, verbose = 1, ))

for _ in range(0):
    net.train(train_x,train_y, epochs = 2, verbose = 1,\
        loss = losses[loss_], optimizer= 'adadelta', batch_size = 16,\
        callback = callback, val_split = 0)
    net.network.compile(optimizer = 'adadelta', loss = RMSError)
    print('RMSE validation error: ', net.network.evaluate(val_x, val_y, verbose = 1, ))


    loss_ = not loss_
    save(model_name)

# TRAIN THE AUTOENCODER NORMALLY
for _ in range(10):
    net.train(train_x,train_y, epochs = 1, verbose = 1,\
            loss = RMSError, optimizer= 'adadelta',\
            batch_size = 32, callback = callback)

    net.network.compile(optimizer = 'adadelta', loss = RMSError)
    print('RMSE validation error: ', net.network.evaluate(val_x, val_y, verbose = 1, ))

    save(model_name)



