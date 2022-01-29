#!/usr/bin/env python
# coding: utf-8

# # Chasing Harry Potter modelling with CNN

# In[1]:


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

import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import fastai

from sklearn.model_selection import train_test_split
from tensorflow .keras.models import Sequential 
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow .keras.layers import Flatten,Dropout, Dense
from keras.preprocessing.image import  load_img,img_to_array
from tensorflow.keras import models as tf_models



# In[2]:


dr = '/Users/ewrimmm/Desktop/ML2/movie_characters/'
harry_dr = "harry-potter/"


# In[3]:


image = load_img(dr+harry_dr + "0001/001.jpg")
plt.imshow(image)
plt.axis("off")
plt.show()


# In[4]:


measure=img_to_array(image)
print(measure.shape)


# # Prepare the data

# index_dr = pd.read_csv(dr+'index.csv');
# 
# index_dr.head(5)
# 

# In[6]:


test_dr = pd.read_csv(dr+'test.csv');


# In[7]:


test_dr.head(5)


# In[8]:


index_dr.drop('Unnamed: 0', axis=1, inplace=True)


# In[9]:


index_dr['name']=None


# In[10]:


index_dr.head()


# In[11]:


index_dr.shape


# In[12]:


index_dr['class_id'].value_counts()


# In[13]:


meta_dr = pd.read_csv(dr+'metadata.csv')


# In[14]:


meta_dr.head(5)


# In[15]:


for i, name in zip(meta_dr['class_id'],meta_dr['minifigure_name']):
    for sor, j in enumerate(index_dr['class_id']):
        if i==j:
            index_dr.iat[sor, 3]=name


# In[16]:


index_dr.tail(10)


# In[17]:


data = pd.merge(index_dr, meta_dr[['class_id', 'minifigure_name']], on='class_id')

data.head(15)


# In[18]:


valid = index_dr.copy()


# In[19]:


sampler = data.sample(20)
sampler.head(5)


# In[20]:


plt.figure(figsize=(16, 10))
for ind, el in enumerate(data.sample(15).iterrows(), 1):
    plt.subplot(3, 5, ind)
    image = cv2.imread(os.path.join(dr, el[1]['path']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f"{el[1]['class_id']}: {el[1]['minifigure_name']}")
    plt.xticks([])
    plt.yticks([])


# In[21]:


plt.figure(figsize=(16, 10))
for ind, el in enumerate(data[data['minifigure_name']=='HARRY POTTER'].sample(10).iterrows(), 1):
    plt.subplot(3, 5, ind)
    image = cv2.imread(os.path.join(dr, el[1]['path']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])


# In[24]:


index_dr.info()


# In[25]:


data.info()


# **Data Augmentation**

# In[26]:


def get_train_transforms():
    return alb.Compose(
        [
            alb.Rotate(limit=30, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            alb.Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=0, p=0.5),
            alb.Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=1, p=0.5),
            alb.HorizontalFlip(p=0.5),
            alb.RandomContrast(limit=(-0.3, 0.3), p=0.5),
            alb.RandomBrightness(limit=(-0.4, 0.4), p=0.5),
            alb.Blur(p=0.25),
        ], 
        p=1.0
    )


# # **Splitting the data for test and train**

# In[27]:


# Get only train rows
tmp_train = data[data['train-valid'] == 'train']
# Get train file paths
train_paths = tmp_train['path'].values
# Get train labels
train_targets = tmp_train['class_id'].values
# Create full train paths (base dir + concrete file)
train_paths = list(map(lambda x: os.path.join(dr, x), train_paths))


# In[28]:


# Get only valid rows
tmp_valid = data[data['train-valid'] == 'valid']
# Get valid file paths
valid_paths = tmp_valid['path'].values
# Get valid labels
valid_targets = tmp_valid['class_id'].values
# Create full valid paths (base dir + concrete file)
valid_paths = list(map(lambda x: os.path.join(dr, x), valid_paths))


# In[29]:


train_data = np.zeros((tmp_train.shape[0], 224, 224, 3))

for i in range(tmp_train.shape[0]):
    
    image = cv2.imread('/Users/ewrimmm/Desktop/ML2/movie_characters/' + tmp_train["path"].values[i])
    
    #Converting BGR to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Resizing image to (512 x 512)
    image = cv2.resize(image, (224,224))
    
    #Normalizing pixel values to [0,1]
    train_data[i] = image / 255.0


# In[30]:


valid_data = np.zeros((tmp_valid.shape[0], 224, 224, 3))

for i in range(tmp_valid.shape[0]):
    
    image = cv2.imread('/Users/ewrimmm/Desktop/ML2/movie_characters/' + tmp_valid["path"].values[i])
    
    #Converting BGR to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Resizing image to (512 x 512)
    image = cv2.resize(image, (224,224))
    
    #Normalizing pixel values to [0,1]
    valid_data[i] = image / 255.0


# In[31]:


valid_label = np.array(tmp_valid["class_id"])-1

train_label = np.array(tmp_train["class_id"])-1


# In[32]:



print('Train Label: ',train_label.shape)
print('Train Data: ',train_data.shape)
print('Valid Data: ',valid_data.shape)
print('Valid Label: ',valid_label.shape)


# In[33]:


from keras.preprocessing.image import ImageDataGenerator


# In[34]:


batch= 15
size= 256
nb_classes=30
IN_SHAPE=(size,size,3)


# In[35]:


train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=20, width_shift_range=0.4, 
                                   height_shift_range=0.4,fill_mode="nearest", zoom_range=0.4, vertical_flip=True, horizontal_flip=True, brightness_range=[0.2,1.0])
valid_datagen = ImageDataGenerator(rescale=1.0/255)


# In[79]:



train_generator = train_datagen.flow_from_dataframe(dataframe=index_dr, directory=dr,
                                                   x_col='path', y_col='name', batch_size= batch,
                                                   shuffle=True, target_size=(size,size))

valid_generator = valid_datagen.flow_from_dataframe(dataframe=valid, directory=dr,
                                                   x_col='path', y_col='name', batch_size= batch,shuffle=False, target_size=(size,size))
             


# # Model Building

# In[210]:


dense_net = tf.keras.applications.DenseNet121()


# In[211]:


dense_net_layer=Dropout(0.5)(dense_net.layers[-2].output)
number_of_classes = len(index_dr['class_id'].unique())


# In[212]:


last_layer = Dense(number_of_classes, activation="softmax")(dense_net_layer)
model = Model(dense_net.input, last_layer)


# In[213]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy'])


# In[214]:


print(model.summary())


# In[219]:


checkpoint = ModelCheckpoint(filepath='model_best', monitor="val_accuracy", save_best_only=True, verbose=1)


# In[220]:


hist=model.fit(
    train_data, 
    train_label, 
    epochs=50, 
    validation_data=(valid_data, valid_label), 
    shuffle=True, 
    batch_size=4, 
    callbacks=checkpoint
)


# In[236]:


acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


# In[237]:


plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()


# ## To see more detailed

# In[221]:


loss_results = pd.DataFrame(model.history.history)


# In[222]:


loss_results.head()


# In[225]:


loss_results[['loss','val_loss']].plot()


# In[227]:


loss_results[['accuracy','val_accuracy']].plot()


# In[228]:


loss_results.plot()


# # Testing the model

# In[234]:


model = tf_models.load_model('model_best')


# In[235]:


sample_df=data.sample(50)


test, _ = train_test_split(sample_df, test_size=0.5)


# In[233]:


test


# In[238]:


for i in range(15):
    
    image = cv2.imread('/Users/ewrimmm/Desktop/ML2/movie_characters/'+test['path'].values[i])
    image = cv2.resize(image, dsize=(224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255
    plt.imshow(image)
    plt.xlabel(test['minifigure_name'].values[i]+'--'+str(test['class_id'].values[i]))
    image = np.reshape(image, (1, 224, 224, 3))
    ans = model.predict(image).argmax()
    ans = ans+1
    minifigure = meta_dr["minifigure_name"][meta_dr["class_id"] == ans].iloc[0]
    print("Class:", str(ans)+ " Minifigure:",minifigure)
    plt.show()


# Predicted Ron Weasley!

# **Before to save some sample for testing, I tried to create dictionary to divide alias and prediction columns. 
# So I could have been matched results, but unfortunately I got key error due to insufficient data**

# # Conclusion

# Now, as we can see, I've increased train accuracy by a large proportion with test accuracy, which is far better than my previus model, but as you can see, the accuracy from training and validation differs significantly, and the losses in my validation set are increasing. Which means my model is overfitting. For those issues, I need more photos to fix this overfitting or change the model that I used below. Or I could also find more rich data to use the same model for better results.
# 
