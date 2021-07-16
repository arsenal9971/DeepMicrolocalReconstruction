## Script to perform WF inpainting using unet as predicting model

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## Import the needed modules
from data.data_factory import generate_data_WFinpaint, DataGenerator_WFinpaint
from ellipse.ellipseWF_factory import plot_WF

import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
import odl
import matplotlib.pyplot as plt

## Load model

# Tensorflow and seed
seed_value = 0
import random
random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)

# Importing relevant keras modules
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from shared.shared import create_increasing_dir
import pickle

# Import model and custom losses
from models.unet import UNet
from models.losses import CUSTOM_OBJECTS

# Parameters for the training
learning_rate = 1e-3
loss = 'mae'
batch_size = 86
epoches = 10000

pretrained = 0
#path_to_model_dir = './models/unets_WFinpaint/training_0'

# Data generator
size = 256
nClasses = 180
lowd = 40
train_gen = DataGenerator_WFinpaint(batch_size, size, nClasses, lowd)
val_gen = DataGenerator_WFinpaint(batch_size, size, nClasses, lowd)

if pretrained==0:
    # Create a fresh model
    print("Create a fresh model")
    unet = UNet()
    model = unet.create_model( img_shape = (size, size, 1) , loss = loss, learning_rate = learning_rate)
    path_to_training = create_increasing_dir('./models/unets_WFinpaint', 'training')
    print("Save training in {}".format(path_to_training))
    path_to_model_dir = path_to_training

else:
    print("Use trained model as initialization:")
    print(path_to_model_dir+"/weights.hdf5")
    model = load_model(path_to_model_dir+"/weights.hdf5",
                       custom_objects=CUSTOM_OBJECTS)
    path_to_training = path_to_model_dir

# Callbacks for saving model
context = {
    "loss": loss,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "path_to_model_dir": path_to_model_dir,
}
path_to_context = path_to_training+'/context.log'

with open(path_to_context, 'wb') as dict_items_save:
    pickle.dump(context, dict_items_save)
print("Save training context to {}".format(path_to_context))

# Save architecture
model_json = model.to_json()
path_to_architecture = path_to_training + "/model.json"
with open(path_to_architecture, "w") as json_file:
    json_file.write(model_json)
print("Save model architecture to {}".format(path_to_architecture))

# Checkpoint for trained model
checkpoint = ModelCheckpoint(
    path_to_training+'/weights.hdf5',
    monitor='val_loss', verbose=1, save_best_only=True)
csv_logger = CSVLogger(path_to_training+'/training.log')

callbacks_list = [checkpoint, csv_logger]

## Training
model.fit_generator(train_gen,epochs=epoches, steps_per_epoch=5600 // batch_size,
                    callbacks=callbacks_list, validation_data=val_gen, validation_steps= 2000// batch_size)
