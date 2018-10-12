import time
import argparse
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import sklearn.metrics as metric

from skimage import io
from skimage import transform
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib


parser = argparse.ArgumentParser(description='Ship Detection')
parser.add_argument('-train_csv', dest='train_csv', type=str, required=True)
parser.add_argument('-model', dest='model_file', type=str, required=False)

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

    # Compute precision
    precision = metric.precision_score(y, y_pred, average=None)

    # Compute recall
    recall = metric.recall_score(y, y_pred, average=None)

    # Compute F1 score
    f1_score = metric.f1_score(y, y_pred, average=None)

    # Compute confusion matrix
    confusion = metric.confusion_matrix(y, y_pred)

    # Plot Confusion matrix
    confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]  # normalize
    df_cm = pd.DataFrame(confusion, index=[i for i in "01"], columns=[i for i in "01"])
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    fig.savefig('cm_validation.png')

    # Print metrics
    print('- Accuracy: ', accuracy)  # show accuracy score
    print('- Precision: ', precision)  # show precision score
    print('- Recall: ', recall)  # show recall score
    print('- F1 Score: ', f1_score)  # show F1 score

    # Save predicted result
    out_df = pd.DataFrame(y_pred, columns=['Output'])
    target_df = pd.DataFrame(y, columns=['Target'])
    
    images = valid_set.iloc[:, 0]
    images.reset_index(drop=True, inplace=True)
    output = pd.concat([images, out_df, target_df], axis=1, ignore_index=True)
    output.columns = ['ImageId', 'Output', 'Target']

    # Save csv
    output.to_csv('output.csv', index=False)


def main():
    args = parser.parse_args()

    # Load dataset
    train_set, valid_set = init_dataset(args.train_csv)

    if args.model_file is None:
        # Start time
        start = time.time()
        
        # Train model
        model = train_model(train_set, valid_set)

        # Elapsed time
        end = time.time()
        print('Elapsed training time: ' + str(end - start) + ' seconds')
    else:
        # Load model
        model = joblib.load(args.model_file)

    # Test model on validation dataset
    validation(model, valid_set)


if __name__ == '__main__':
    main()
