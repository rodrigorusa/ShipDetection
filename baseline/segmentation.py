import argparse
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.metrics as metric

from skimage import io
from skimage import transform
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib


parser = argparse.ArgumentParser(description='Ship Detection Segmentation')
parser.add_argument('-train_csv', dest='train_csv', type=str, required=True)
parser.add_argument('-images', dest='images_folder', type=str, required=True)
parser.add_argument('-fraction', dest='fraction', type=float, required=True)

TRAINING_FRAC = 0.8

def init_dataset(train_csv, fraction):
    print('Initializing dataset...')

    # Read csv
    df_train = pd.read_csv(train_csv)
    
    # Select a subset of training samples
    df_sub_train = df_train.sample(frac=fraction, random_state=1)

    # Select training samples
    train_set = df_sub_train.sample(frac=TRAINING_FRAC, random_state=1)   

    print('Number of training samples: ', train_set.shape[0])

    # Select validation samples
    valid_set = df_sub_train.drop(train_set.index)

    print('Number of validation samples: ', valid_set.shape[0])

    return train_set, valid_set

def get_mask(encoded_pixels):
    mask = np.zeros((768*768), np.uint8)

    pixels = str(encoded_pixels)
    if pixels != 'nan':
        array = pixels.split(' ')
        
        for i in range(0, len(array), 2):
            pixel = int(array[i])
            step = int(array[i+1])
            for j in range(0, step):
                mask[pixel + j] = 255

        #plt.imshow(mask.reshape((768,768)), cmap='gray')
        #plt.show()

    mask = mask.reshape((768,768))
    mask_t = transform.downscale_local_mean(mask, (2, 2))

    return mask_t.reshape((-1))


def train_model(train_set, images_folder, valid_set):
    print('Start training model...')

    # Number of training samples
    m = train_set.shape[0]

    print('Number of samples: ', m)

    # Create empty dataframe
    x = np.empty((0, 64*64*3), float)

    y = np.empty((0, 384*384), float)

    for i in range(m):
        #print(train_set.iloc[i, 0])

        # Read image
        image_path = images_folder + '/' + train_set.iloc[i, 0]
        img = io.imread(image_path)

        # Downscale
        img_t = transform.downscale_local_mean(img, (12, 12, 1))
        img_t = img_t.astype(np.uint8)

        #plt.imshow(img_t.astype(np.uint8))
        #plt.show()
        #exit(0)

        img_norm = img_t.astype(float)/255.0

        # Append to dataframe
        rgb = np.reshape(img_norm, (-1, 3))
        array = np.concatenate((rgb[:, 0], rgb[:, 1], rgb[:, 2]))
        x = np.vstack((x, array))

        mask = get_mask(train_set.iloc[i, 1])

        y = np.vstack((y, mask))


    # Setup the classifier
    model = MLPClassifier(hidden_layer_sizes=(4096, 2048), activation='relu', max_iter=1000, learning_rate_init=0.001, verbose=10)
    
    # Training the model
    print('Training model...')
    model.fit(x, y)

    # Save the model on hard disk for future uses
    joblib.dump(model, 'model.joblib')

    # Plot loss
    fig = plt.figure()
    plt.plot(model.loss_curve_)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
    fig.savefig('error_plot.png')

    return model

    
def validation(model, valid_set):
    print('Start model validation...')

    # Number of validation samples
    m = valid_set.shape[0]

    # Validation targets
    y = valid_set.iloc[:, 1].values

    # Validation input
    x = valid_set.iloc[:, 2:].values

    # Normalize
    x = x/255.0

    # Predict labels
    print('Validation:')
    y_pred = model.predict(x)

    # Compute accuracy
    accuracy = metric.accuracy_score(y, y_pred, normalize=True)
    print('- Accuracy: ', accuracy)  # show accuracy score
    

def main():
    args = parser.parse_args()

    # Modify input
    train_set, valid_set = init_dataset(args.train_csv, args.fraction)

    # Train model
    model = train_model(train_set, args.images_folder, valid_set)

    # Test model on validation dataset
    #validation(model, valid_set)


if __name__ == '__main__':
    main()