import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories

from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks.

TRAIN_BAC_DIR = 'train/bac_blight'
TRAIN_CER_DIR = 'train/cercospora_leaf_spot'
TRAIN_AL_DIR = 'train/alternaria_alternata'
TRAIN_ANTH_DIR = 'train/anthracnose'
TRAIN_HEALTHY_DIR = 'train/healthy'

IMG_SIZE = 72
LR = 1e-3
MODEL_NAME = 'leafDectection-{}.model'.format(LR)

# if you write the code seperately, the code will not be dry enough
def create_train_data():
    training_data = []
    categories = {TRAIN_HEALTHY_DIR: [1, 0, 0, 0, 0],
                TRAIN_BAC_DIR: [0, 1, 0, 0, 0],
                TRAIN_CER_DIR: [0, 0, 1, 0, 0],
                TRAIN_AL_DIR: [0, 0, 0, 1, 0],
                TRAIN_ANTH_DIR: [0, 0, 0, 0, 1]}
    for cat in categories.keys():
        for img in tqdm(os.listdir(cat)):
            label = categories[cat];
            path = os.path.join(cat, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            try:
                if img.size > 0:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            except AttributeError as e:
                print(str(path))

            training_data.append([np.array(img), np.array(label)])

    np.save('train_data.npy', training_data)
    return training_data

train_data = create_train_data()

# If you have already created the dataset:
# train_data = np.load('train_data.npy')


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

tf.reset_default_graph() # Clears the default graph stack and resets the global default graph.

network = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3])
# IMG_SIZE = 72, LR = 1e-3
network = conv_2d(network, 64, 3, strides=3, activation='relu')
network = max_pool_2d(network, 3, strides=2)

network = conv_2d(network, 128, 3, strides=3, activation='relu')
network = max_pool_2d(network, 3, strides = 2)

network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 1, activation='relu')

network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)

network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 1024, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=LR)
# LR = 1e-3

# Training
model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir="logs")

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

# reshape train set
X = np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train_data]

model.fit(X, Y, n_epoch=60, validation_set=0.1, shuffle=True, show_metric=True,
    batch_size=64, snapshot_step=50, snapshot_epoch=False, run_id=MODEL_NAME)

model.save(MODEL_NAME)
