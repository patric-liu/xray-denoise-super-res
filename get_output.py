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

# Load Model
model_name = 'model1'

# model reconstruction from JSON
net = network.Network(None, None, None, test = 0)
with open(os.getcwd() + '/model_weights/'+ model_name +'_architecture.json', 'r') as f:
    net.network = model_from_json(f.read())
# load weights into the model
net.network.load_weights(os.getcwd() + '/model_weights/'+ model_name +'_weights.h5')

net.network.summary()


# load data
flatten = False      # Get 1D data
rescale = True       #  Rescale 'RGB' values from [0,255] to [1,0]
denoise_only = False # Reshape target values from 128x128 to 64x64
test_data, rng = getdata.get_test(flatten = flatten, rescale = rescale,\
                              denoise_only = denoise_only, amount = 3999)

# save data
for n, index in enumerate(rng):
    test_output = net.predict(np.array([test_data[n], test_data[n]]))[0]
    test_output = (1-test_output)*255
    path = os.getcwd() + '/xray_images/test_images_128x128/test_' + ('00000' + str(index))[-5:] + '.png'
    cv2.imwrite(path,test_output,[cv2.IMWRITE_PNG_COMPRESSION, 0])
    if n%200 == 0:
        print("finished processing {} images!".format(n))
        
print('done')

