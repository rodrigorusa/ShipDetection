# ShipDetection
Ship Detection using Deep Learning and Machine Learning techniques

## Generate input

### Images 64x64 RGB

To generate a subset of complete dataset run:

python3 generate_input.py -csv_file <train_csv_original> -images <train_images_folder> -fraction <factor_of_samples>

### DCT of images 128x128 (1 channel)

To generate a subset of complete dataset run:

python3 generate_input_dct.py -csv_file <train_csv_original> -images <train_images_folder> -fraction <factor_of_samples>

## Training model

To train and validate your model run:

python3 main.py -train_csv <csv_generated> [-model_file <pre_trained_model>]

The model_file is optional, if you pass the model the script will validate only. Otherwise the script will train the model and validate.