## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## Original copyright license follows.

import tensorflow as tf
import numpy as np
import os
import threading
import time
import random
import pickle
import gzip
import urllib.request
import matplotlib.pyplot as plt

from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.models import load_model

def extract_data(filename, num_images):
	with gzip.open(filename) as bytestream:
		bytestream.read(16)
		buf = bytestream.read(num_images*28*28)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		data = (data / 255) - 0.5
		data = data.reshape(num_images, 28, 28, 1)
		return data

def extract_labels(filename, num_images):
	with gzip.open(filename) as bytestream:
		bytestream.read(8)
		buf = bytestream.read(1 * num_images)
		labels = np.frombuffer(buf, dtype=np.uint8)
	return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
	def __init__(self, folder_path=None):

		data_path = os.path.join(folder_path,"mnist_data")
		if not os.path.exists(data_path):
			os.mkdir(data_path)
			files = ["train-images-idx3-ubyte.gz",
					 "t10k-images-idx3-ubyte.gz",
					 "train-labels-idx1-ubyte.gz",
					 "t10k-labels-idx1-ubyte.gz"]
			for name in files:
				urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, f"{data_path}/"+name)

		train_data = extract_data(f"{data_path}/train-images-idx3-ubyte.gz", 60000)
		train_labels = extract_labels(f"{data_path}/train-labels-idx1-ubyte.gz", 60000)
		self.test_data = extract_data(f"{data_path}/t10k-images-idx3-ubyte.gz", 10000)
		self.test_labels = extract_labels(f"{data_path}/t10k-labels-idx1-ubyte.gz", 10000)
		
		VALIDATION_SIZE = 5000
		
		self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
		self.validation_labels = train_labels[:VALIDATION_SIZE]
		self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
		self.train_labels = train_labels[VALIDATION_SIZE:]


class MNISTModel:
	def __init__(self, restore = None, session=None, use_softmax=False):
		self.num_channels = 1
		self.image_size = 28
		self.num_labels = 10

		model = Sequential()

		model.add(Conv2D(32, (3, 3),
						 input_shape=(28, 28, 1)))
		model.add(Activation('relu'))
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
		model.add(Flatten())
		model.add(Dense(200))
		model.add(Activation('relu'))
		model.add(Dense(200))
		model.add(Activation('relu'))
		model.add(Dense(10))
		# output log probability, used for black-box attack
		if use_softmax:
			model.add(Activation('softmax'))
		if restore:
			model.load_weights(restore)

		self.model = model

	def predict(self, data):
		return self.model(data)

class MNISTHuman:
	def __init__(self, data):
		self.num_channels = 1
		self.image_size = 28
		self.num_labels = 10
		self.data = data

		self.create_catwise_dataset()
		self.training_trials()

	def create_catwise_dataset(self):
		self.catwise_dataset = dict([(i, []) for i in range(self.num_labels)])
		for i, l in enumerate(self.data.test_labels):
			self.catwise_dataset[l.argmax()].append(i)

	def ooo_trial(self, reg3_ims, odd1_im):
		fig, ax = plt.subplots(3,3)
		poscs = [(0,1),(1,0),(1,2),(2,1)]

		odd1_pos = np.random.randint(0, 4)
		odd1_posc = poscs[odd1_pos]

		ax[odd1_posc[0], odd1_posc[1]].imshow(odd1_im, cmap='gray')

		poses = list(range(4))
		poses.remove(odd1_pos)
		for i, posc in enumerate([poscs[p] for p in poses]):
			ax[posc[0], posc[1]].imshow(reg3_ims[i], cmap='gray')

		for axi in ax.flatten():
			axi.axis('off')

		# Create a function to handle mouse clicks
		def blank_figure():
			time.sleep(2)  # Wait for 1 second
			for axi in ax.flatten():
				axi.clear()
				axi.set_facecolor('white')
				axi.axis('off')
			fig.canvas.draw()

		def on_numberkey_press(event):
			if event.key.isdigit():
				self.trial_confidence = (int(event.key) - 1) / 8

				plt.close(fig)

		def on_arrowkey_press(event):
			key_map = ['up', 'left', 'right', 'down']
			if event.key in key_map:
				self.trial_response = key_map.index(event.key)
				self.trial_correct = (self.trial_response == odd1_pos)

		# Connect the click event to the function
		fig.canvas.mpl_connect('key_press_event', on_arrowkey_press)
		fig.canvas.mpl_connect('key_press_event', on_numberkey_press)

		# Start a thread to close the figure after 1 second
		thread = threading.Thread(target=blank_figure)
		thread.start()

		plt.show()

		print(f"{self.trial_correct}. Confidence: {self.trial_confidence*100}%")

	def add_salt_pepper_noise(self, im, amount=0.32, salt_vs_pepper=0.5):
		"""
		Add salt and pepper noise to an image.

		Parameters:
		- image: input image as a numpy array.
		- amount: fraction of image pixels to be affected.
		- salt_vs_pepper: proportion of salt vs. pepper noise.
		"""
		# Create noise mask
		rows, cols = im.shape
		num_salt = np.ceil(amount * im.size * salt_vs_pepper)
		num_pepper = np.ceil(amount * im.size * (1.0 - salt_vs_pepper))

		# Add salt noise
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in (rows, cols)]
		im[coords[0], coords[1]] = 1

		# Add pepper noise
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in (rows, cols)]
		im[coords[0], coords[1]] = 0

		return im

	def add_noise(self, im):
		return 0.1 * im + 0.2 * np.random.uniform(0, 1, size=(28,28))

	def training_trials(self, n=3):
		for i in range(n):
			reg_cat = np.random.randint(0, 10)
			odd_cat = np.random.randint(0, 10)

			reg3 = random.sample(self.catwise_dataset[reg_cat], 3)
			odd1 = random.choice(self.catwise_dataset[odd_cat])

			# add noise to images
			reg3_ims = []
			for i in reg3:
				im = self.data.test_data[i].reshape(28,28)
				# reg3_ims.append(self.add_salt_pepper_noise(im))
				reg3_ims.append(self.add_noise(im))
			# odd1_im = self.add_salt_pepper_noise(self.data.test_data[odd1].reshape(28,28))
			odd1_im = self.add_noise(self.data.test_data[odd1].reshape(28,28))

			self.ooo_trial(reg3_ims, odd1_im)

	def predict(self, data):
		if self.rt_min is None or self.rt_max is None:
			raise ValueError("RT min and max are both None. Can't estimate confidence.")
			return
		
		return 1