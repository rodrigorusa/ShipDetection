import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metric

from skimage import io
from skimage import transform
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib


parser = argparse.ArgumentParser(description='Ship Detection')
parser.add_argument('-train_csv', dest='train_csv', type=str, required=True)

TRAINING_FRAC = 0.8

def init_dataset(train_csv):
    print('Initializing dataset...')

    # Read csv
    df_train = pd.read_csv(train_csv)
    
    # Select training samples
    train_set = df_train.sample(frac=TRAINING_FRAC, random_state=1)   

    print('Number of training samples: ', train_set.shape[0])

    # Select validation samples
    valid_set = df_train.drop(train_set.index)

    print('Number of validation samples: ', valid_set.shape[0])

    return train_set, valid_set

def train_model(train_set, valid_set):
    print('Start training model...')

    # Number of training samples
    m = train_set.shape[0]

    # Training targets
    y = train_set.iloc[:, 1].values

    # Training input
    x = train_set.iloc[:, 2:].values

    # Normalize
    x = x/255.0

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
    train_set, valid_set = init_dataset(args.train_csv)

    # Train model
    model = train_model(train_set, valid_set)

    # Test model on validation dataset
    validation(model, valid_set)


if __name__ == '__main__':
    main()
