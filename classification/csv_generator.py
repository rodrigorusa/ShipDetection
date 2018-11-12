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

parser = argparse.ArgumentParser(description='CSV Generator')
parser.add_argument('-csv', dest='csv_file', type=str, required=True)
parser.add_argument('-fraction', dest='fraction', type=float, required=True)


def generator(csv_file, fraction):
    print("Generating csv...")

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
    df_set = df_dropped.sample(frac=fraction, random_state=1)

    m = df_set.shape[0]
    print("Number of samples: ", m)

    # Save csv
    df_set.to_csv('train_modified_' + str(m) + '.csv', index=False)

    print("Done!")

def main():
    args = parser.parse_args()

    # Generate csv
    generator(args.csv_file, args.fraction)

if __name__ == '__main__':
    main()
