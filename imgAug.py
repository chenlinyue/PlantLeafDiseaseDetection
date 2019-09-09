#!/usr/bin/env python
#-*- coding: utf-8 -*-

# import packages
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from tqdm import tqdm


datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=[0.1, 0.2, 0.12, 0.08, 0.15, 0],
        height_shift_range=[0.1, 0.2, 0.12, 0.08,0.15, 0],
        shear_range=0.3,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='constant',
        cval = 0)

TRAIN_BAC_DIR = '/original_train/bac_blight'
TRAIN_CER_DIR = '/original_train/cercospora_leaf_spot'
TRAIN_AL_DIR = '/original_train/alternaria_alternata'
TRAIN_ANTH_DIR = '/original_train/anthracnose'
TRAIN_HEALTHY_DIR = '/original_train/healthy'

for img in tqdm(os.listdir(TRAIN_HEALTHY_DIR)):
  img_mark = img.split('.')[0]
  path = os.path.join(TRAIN_HEALTHY_DIR, img)
  img = load_img(path)
  x = img_to_array(img)
  x = x.reshape((1, ) + x.shape)

  i = 0
  # in case of original data size change the factor for each leaf disease type
  for batch in datagen.flow(x, batch_size=1, save_to_dir='/bac_blight',
  save_prefix=img_mark, save_format='jpg'):
    i += 1
    if i >= 10:
      break
