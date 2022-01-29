#!/usr/bin/env python
# coding: utf-8

# # Chasing Harry Potter modelling with Multilayer Perceptron (MLP)

# In[3]:


import os
import math
import matplotlib.pyplot as plt
import random
import cv2
import albumentations as alb
import efficientnet.tfkeras as efn
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model



import numpy as np
import pandas as pd

import seaborn as sn
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow .keras.models import Sequential 
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow .keras.layers import Flatten,Dropout, Dense
from keras.preprocessing.image import  load_img,img_to_array
from tensorflow.keras import models as tf_models

import datetime
import fastai


# In[4]:


dr = '/Users/ewrimmm/Desktop/ML2/movie_characters/'


# In[5]:


index_dr = pd.read_csv(dr+'index.csv');

index_dr.head(5)


# In[6]:


test_dr = pd.read_csv(dr+'test.csv');

test_dr.head(5)


# In[7]:


meta_dr = pd.read_csv(dr+'metadata.csv')
meta_dr.head(5)


# # Training & Testing

# In[8]:


train_df = index_dr.merge(meta_dr, on = 'class_id')
test_df = test_dr.merge(meta_dr, on = 'class_id')


# In[9]:


train_df.head()


# In[10]:


IMG_SIZE = 240

def data_load_def(dataframe, batch_size = 32, img_size = IMG_SIZE, directory_path = dr, rescale = True):

    if rescale:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator()


    tf_dataset = datagen.flow_from_dataframe(dataframe,
                                            directory = directory_path,
                                            x_col = 'path',
                                            y_col = 'minifigure_name',
                                            target_size = (img_size, img_size),
                                            batch_size = batch_size,)
    return tf_dataset


# In[11]:


train_ds = data_load_def(train_df)
test_ds = data_load_def(test_df)


# # Modelling

# In[12]:


tf.random.set_seed(42)


# In[13]:


model_mp = tf.keras.Sequential([
        tf.keras.layers.Input(shape = (train_ds.image_shape), name = 'input'),
        tf.keras.layers.Flatten(name= 'flatten'),
        tf.keras.layers.Dense(100, activation='relu', name= 'first_hidd_layer'),
        tf.keras.layers.Dense(64, activation='relu', name= 'second_hidd_layer'),
        tf.keras.layers.Dense(30, activation='softmax', name='output_layer')
], name = 'MLP')


# In[14]:


model_mp.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
             metrics = ['accuracy'])


# In[15]:


model_mp.summary()


# # Create Model Callbacks

# In[16]:




def tb_callback_def(dir_name, experiment_name):

    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback

checkpoint_path = "model_checkpoints/mlp.ckpt"

model_cp = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      monitor="val_acc", # save the model weights with best validation accuracy
                                                      save_best_only=True, # only save the best weights
                                                      save_weights_only=True, # only save model weights (not whole model)
                                                      verbose=0)


# In[19]:


tf.random.set_seed(42)


hist_model_mp = model_mp.fit(
    train_ds,
    epochs = 12,
    steps_per_epoch = len(train_ds),
    validation_data = test_ds,
    validation_steps= len(test_ds),
    callbacks=[tb_callback_def('training_logs','lego_mlp'),model_cp])


# # Evaluate the performance of the model

# In[20]:


evol_model = model_mp.evaluate(test_ds)
evol_model


# In[21]:


acc = hist_model_mp.history['accuracy']
val_acc = hist_model_mp.history['val_accuracy']
loss = hist_model_mp.history['loss']


val_loss = hist_model_mp.history['val_loss']
epochs = range(len(hist_model_mp.history['loss']))


plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


# In[22]:


plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()


# # Conclusion

# For the first plot, we can see the result of accuracy- training value is around 0.045 which is fine
# and the validation value is around 0.068.
# 
# I tend to improve data-set prediction accuracy by increasing or decreasing data feature or feature selection, 
# or by incorporating feature engineering into our machine learning model.
# 
# For the second plot, we have losses in out model which is underfitting for our sample. 
# Predicting and clasifying our data also mean depending on well-fitted model&data. 
# 

# In[ ]:




