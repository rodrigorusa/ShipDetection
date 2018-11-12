import time
import argparse
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from sklearn.metrics import classification_report, confusion_matrix

from generator import RGB_Generator
from generator import DCT_Generator

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-csv', dest='csv_file', type=str, required=True)
parser.add_argument('-images', dest='images_folder', type=str, required=True)
parser.add_argument('-image_size', dest='image_size', type=int, required=True)
parser.add_argument('-batch_size', dest='batch_size', type=int, required=True)
parser.add_argument('-epochs', dest='epochs', type=int, required=True)


TRAINING_FRAC = 0.8

def init_dataset(csv_file):

	# Read dataset
    df = pd.read_csv(csv_file)

    # Select training samples
    train_set = df.sample(frac=TRAINING_FRAC, random_state=1)   

    print('[INFO] Number of training samples: ', train_set.shape[0])

    # Select validation samples
    valid_set = df.drop(train_set.index)

    print('[INFO] Number of validation samples: ', valid_set.shape[0])

    return train_set.values, valid_set.values

def create_model(image_size):

	# Neural Network (4096, 2048)
	# model = Sequential([
 #        Dense(4096, input_shape=(image_size*image_size,)),
 #        Activation('relu'),
 #        Dense(2048, input_shape=(4096,)),
 #        Activation('relu'),
 #        Dense(1),
 #        Activation('sigmoid')
 #    ])

	# RGB Convolutional Neural Network
	model = Sequential([
		Conv2D(32, (3, 3), input_shape=(image_size, image_size, 3)),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Conv2D(32, (3, 3)),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Conv2D(64, (3, 3)),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2)),
		Flatten(),	# this converts our 3D feature maps to 1D feature vectors
		Dense(64),
		Activation('relu'),
		Dropout(0.5),
		Dense(1),
		Activation('sigmoid')
	])

	return model

def plot_curves(history):
	# Loss Curves
	fig = plt.figure(figsize=(8, 6))
	plt.plot(history.history['loss'],'r')
	plt.plot(history.history['val_loss'],'b')
	plt.legend(['Training loss', 'Validation Loss'])
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.title('Loss Curves')
	fig.savefig('loss_curve.png')
 
	# Accuracy Curves
	fig = plt.figure(figsize=(8, 6))
	plt.plot(history.history['acc'],'r')
	plt.plot(history.history['val_acc'],'b')
	plt.legend(['Training Accuracy', 'Validation Accuracy'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.title('Accuracy Curves')
	fig.savefig('acc_curve.png')

def plot_confusion_matrix(gt, pred):

	cm = confusion_matrix(gt, pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize
	df_cm = pd.DataFrame(cm, index=[i for i in "01"], columns=[i for i in "01"])
	fig = plt.figure(figsize=(10, 7))
	sn.heatmap(df_cm, annot=True)
	fig.savefig('cm_validation.png')

def save_predicted(images, gt, pred):

	images = pd.DataFrame(images, columns=['ImageId'])
	out_df = pd.DataFrame(pred, columns=['Output'])
	target_df = pd.DataFrame(gt, columns=['Target'])
    
	output = pd.concat([images, out_df, target_df], axis=1, ignore_index=True)
	output.columns = ['ImageId', 'Output', 'Target']

	# Save csv
	output.to_csv('output.csv', index=False)

def save_model(model):
	
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	
	# Serialize weights to HDF5
	model.save_weights("model.h5")
	print("[INFO] Saved model to disk")


def main():
	args = parser.parse_args()

	# Load csv with image filenames and labels
	train_set, valid_set = init_dataset(args.csv_file)

	# Get parameters
	images_folder = args.images_folder
	image_size = args.image_size
	batch_size = args.batch_size
	epochs = args.epochs

	# Define generator data class
	train_generator = RGB_Generator(train_set[:,0], images_folder, train_set[:,1], image_size, batch_size)
	valid_generator = RGB_Generator(valid_set[:,0], images_folder, valid_set[:,1], image_size, batch_size)

	# Create model
	model = create_model(image_size)
	print(model.summary())

	# Compile
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	# Train the model
	print('[INFO] Training model...')
	start = time.time()
	history = model.fit_generator(	generator=train_generator,
									steps_per_epoch=(train_set.shape[0] // batch_size),
									epochs=epochs,
									validation_data=valid_generator,
									validation_steps=(valid_set.shape[0] // batch_size))
	end = time.time()
	print('[INFO] Elapsed training time: %.2f seconds' % (end - start))

	# Plot loss and accuracy curves
	plot_curves(history)

	# Compute the training accuracy
	score = model.evaluate_generator(train_generator, workers=8, use_multiprocessing=True)
	print('[INFO] Training accuracy: %.2f%%' % (score[1]*100))

	# Predict (validation)
	print('[INFO] Validating model...')
	y_pred = model.predict_generator(valid_generator, steps=(valid_set.shape[0] // batch_size+1), verbose=1)
	y_pred = np.array([int(round(x[0])) for x in y_pred])
	
	# Compute the validation accuracy
	score = model.evaluate_generator(valid_generator, workers=8, use_multiprocessing=True)
	print('[INFO] Validation accuracy: %.2f%%' % (score[1]*100))

	# Plot confusion matrix
	y = valid_set[:,1].astype('int64')
	plot_confusion_matrix(y, y_pred)

	# Generate report of metrics
	target_names = ['0', '1']
	print('[INFO] Metrics Report:')
	print(classification_report(y, y_pred, target_names=target_names))

	# Save predicted result
	save_predicted(valid_set[:, 0], y, y_pred)

	# Serialize model to JSON and save it
	save_model(model)
	

if __name__ == '__main__':
    main()