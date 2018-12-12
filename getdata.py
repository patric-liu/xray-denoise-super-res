import numpy as np 
import cv2
import os

path = os.getcwd() + '/xray_images'
test_size = 3999
train_size = 19999 - test_size

def load(sub_path, rng, array, flatten, rescale, denoise_only):
	for n, index in enumerate(rng):
		full_path = sub_path + ('00000' + str(index))[-5:] + '.png'
		img = img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
		if denoise_only:
			img = cv2.resize(img,(64,64))
		if flatten:
			img = img.flatten()
		else:
			img = np.expand_dims(img,axis=2)
		if rescale:
			img = (1-(img/255))
		array[n] = img
	return array

def get_training(amount, flatten = False, rescale = True, denoise_only = False):
	if amount == 0:
		amount = train_size
	rng = list(range(test_size + 1, test_size + amount + 1))
	data_x = None
	data_y = None
	if flatten:
		data_x = np.zeros((amount,64*64))
		if denoise_only:
			data_y = np.zeros((amount,64*64))
		else:
			data_y = np.zeros((amount,128*128))
	else:
		data_x = np.zeros((amount,64,64,1))
		if denoise_only:
			data_y = np.zeros((amount,64,64,1))
		else:
			data_y = np.zeros((amount,128,128,1))

	data_x = load(path + '/train_images_64x64/train_', rng, data_x, flatten, rescale, denoise_only)
	data_y = load(path + '/train_images_128x128/train_', rng, data_y, flatten, rescale, denoise_only)

	return data_x, data_y


def get_test(amount, flatten = False, rescale = True, denoise_only = False, sample = 0):
	if amount == 0:
		amount = test_size

	print(amount)
	rng = list(range(sample + 1, sample + amount + 1))
	data_x = None
	if not flatten:
		data_x = np.zeros((amount,64,64,1))
	else:
		data_x = np.zeros((amount,64*64))
	data_x = load(path + '/test_images_64x64/test_', rng, data_x, flatten, rescale, denoise_only)
	return data_x, rng
