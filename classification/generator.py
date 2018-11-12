import numpy as np

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean
from keras.utils import Sequence
from scipy import fftpack

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

IMAGE_SIZE = 768

class RGB_Generator(Sequence):

	def __init__(self, image_filenames, image_folder, labels, size, batch_size):
		self.image_filenames = image_filenames
		self.image_folder = image_folder
		self.labels = labels
		self.size = size
		self.batch_size = batch_size

	def __len__(self):
		return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

	def __getitem__(self, idx):
		images = self.image_filenames[idx * self.batch_size:(idx+1) * self.batch_size]
		labels = self.labels[idx * self.batch_size:(idx+1) * self.batch_size]

		# Number of images
		m = len(images)

		# Downscale factor
		down_factor = int(IMAGE_SIZE/self.size)

		# Create empty dataframe
		x = np.empty((0, 128, 128, 3), float)

		for i in range(m):
			# Read image
			image_path = self.image_folder + '/' + images[i]
			img = imread(image_path)

			# Downscale
			img_t = downscale_local_mean(img, (down_factor, down_factor, 1))
			img_t = img_t.astype(np.uint8)

			# Normalize
			img_norm = img_t/255.0

			# Append to dataframe
			img_norm = img_norm.reshape((1,) + img_norm.shape)
			x = np.vstack((x, img_norm))

		return x, labels

class DCT_Generator(Sequence):

	def __init__(self, image_filenames, image_folder, labels, size, batch_size):
		self.image_filenames = image_filenames
		self.image_folder = image_folder
		self.labels = labels
		self.size = size
		self.batch_size = batch_size

	def __len__(self):
		return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

	def __getitem__(self, idx):
		images = self.image_filenames[idx * self.batch_size:(idx+1) * self.batch_size]
		labels = self.labels[idx * self.batch_size:(idx+1) * self.batch_size]

		# Number of images
		m = len(images)

		# Downscale factor
		down_factor = int(IMAGE_SIZE/self.size)

		# Create empty dataframe
		x = np.empty((0, self.size*self.size), float)

		for i in range(m):
			# Read image
			image_path = self.image_folder + '/' + images[i]
			img = imread(image_path)

			# Convert to grayscale
			img_gray = rgb2gray(img)

			# Compute DCT
			dct = fftpack.dct(fftpack.dct(img_gray.T, norm='ortho').T, norm='ortho')
			dct = np.abs(dct)

			# Downscale
			dct_t = downscale_local_mean(dct, (down_factor, down_factor))

			# Append to dataframe
			array = np.reshape(dct_t, (-1))
			x = np.vstack((x, array))

		return x, labels