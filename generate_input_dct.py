import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metric

from skimage import io
from skimage import transform
from skimage.color import rgb2gray
from scipy import fftpack

parser = argparse.ArgumentParser(description='Generate input csv')
parser.add_argument('-csv', dest='csv_file', type=str, required=True)
parser.add_argument('-images', dest='images_folder', type=str, required=True)
parser.add_argument('-fraction', dest='fraction', type=float, required=True)


def generate(csv_file, images_folder, fraction):
    print("Generating input...")

    # Read dataset
    df = pd.read_csv(csv_file)

    # Add label
    df.loc[df['EncodedPixels'].notnull(), 'EncodedPixels'] = 1
    df.loc[df['EncodedPixels'].isnull(), 'EncodedPixels'] = 0

    # Change column name
    df.columns = ['ImageId', 'Label']

    # Remove duplicates
    df_dropped = df.drop_duplicates(['ImageId'], keep='first')

    # Select a fraction of samples
    df_set = df_dropped.sample(frac=fraction)

    m = df_set.shape[0]
    print("Number of samples: ", m)

    # Create empty dataframe
    x = np.empty((0, 128*128), float)

    for i in range(m):
        #print(df_set.iloc[i, 1])

        # Read image
        image_path = images_folder + '/' + df_set.iloc[i, 0]
        img = io.imread(image_path)

        # Convert to grayscale
        img_gray = rgb2gray(img)

        # Compute DCT
        dct = fftpack.dct(fftpack.dct(img_gray.T, norm='ortho').T, norm='ortho')
        dct = np.abs(dct)

        # Downscale
        dct_t = transform.downscale_local_mean(dct, (6, 6))
        
        # Append to dataframe
        array = np.reshape(dct_t, (-1))
        x = np.vstack((x, array))

    # Column names
    column_names = ['ImageId', 'Label']
    for i in range(x.shape[1]):
        column_names.append('Pixel_' + str(i))

    # Create dataframe from pixels values
    df_pixels = pd.DataFrame(x, columns=column_names[2:])

    # Concatenate the pixels values
    df_set.reset_index(drop=True, inplace=True)
    df_set = pd.concat([df_set, df_pixels], axis=1, ignore_index=True)

    # Add column names
    df_set.columns = column_names

    # Save csv
    df_set.to_csv('train_modified.csv', index=False)


def main():
    args = parser.parse_args()

    # Start time
    start = time.time()

    # Generate input
    generate(args.csv_file, args.images_folder, args.fraction)

    # Elapsed time
    end = time.time()
    print('Elapsed time: ' + str(end - start) + ' seconds')

if __name__ == '__main__':
    main()
