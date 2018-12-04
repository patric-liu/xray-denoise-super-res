import numpy as np 
import cv2
import os

path = os.getcwd() + '/xray_images'
test_size = 3999
train_size = 19999 - test_size

'''def get_training(flatten = False, amount = train_size, rescale = True):
	if not flatten:
		data_x = np.zeros((amount,64,64))
		data_y = np.zeros((amount,128,128))
		for n, index in enumerate(list(range(test_size + 1, amount + test_size + 1))):
			x_path = path + '/train_images_64x64/train_' + ('00000' + str(index))[-5:] + '.png'
			y_path = path + '/train_images_128x128/train_' + ('00000' + str(index))[-5:] + '.png'
			data_x[n] = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
			data_y[n] = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
		return data_x,data_y
	elif flatten:
		data_x = np.zeros((amount,64*64))
		data_y = np.zeros((amount,128*128))
		for n, index in enumerate(list(range(test_size + 1, amount + test_size + 1))):
			x_path = path + '/train_images_64x64/train_' + ('00000' + str(index))[-5:] + '.png'
			y_path = path + '/train_images_128x128/train_' + ('00000' + str(index))[-5:] + '.png'
			data_x[n] = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE).flatten()
			data_y[n] = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE).flatten()
		return data_x,data_y'''

def load(sub_path, rng, array, flatten, rescale, denoise_only = False):
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

def get_training(flatten = False, amount = train_size, rescale = True, denoise_only = True):
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

	data_x = load(path + '/train_images_64x64/train_', rng, data_x, flatten, rescale)
	data_y = load(path + '/train_images_128x128/train_', rng, data_y, flatten, rescale, denoise_only)

	return data_x, data_y


def get_test(flatten = False, amount = test_size, rescale = True):
	rng = list(range(1, amount + 1))
	if not flatten:
		data_x = np.zeros((amount,64,64,1))
		data_x = load(path + '/test_images_64x64/test_', rng, data_x, False, rescale)
		return data_x
	else:
		data_x = np.zeros((amount,64*64))
		data_x = load(path + '/test_images_64x64/test_', rng, data_x, True, rescale)
		return data_x
