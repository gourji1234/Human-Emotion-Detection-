#!/usr/bin/env python
# coding: utf-8

# # Model Generation

# In[2]:


import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os


# In[3]:


num_classes=7
img_rows,img_cols=48,48
batch_size=32


# In[4]:


train_data_dir="E:\\Major Project\\Training"
validation_data_dir="E:\\Major Project\\Validation_main"


# In[5]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)


# In[6]:


train_generator = train_datagen.flow_from_directory(
                        train_data_dir,
                        color_mode='grayscale',
                        target_size=(img_rows,img_cols),
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
                                validation_data_dir,
                                color_mode='grayscale',
                                target_size=(img_rows,img_cols),
                                batch_size=batch_size,
                                class_mode='categorical',
                                shuffle=True)


# In[7]:


model = Sequential()


# In[8]:


model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# In[9]:


model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# In[10]:


model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# In[11]:


model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# In[12]:


model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# In[13]:


model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# In[14]:


model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))


# In[15]:


print(model.summary())


# In[16]:


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# In[17]:


checkpoint = ModelCheckpoint("E:\\Major Project\\EmotionDetectionModel.h5.keras",
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]


# In[18]:


from keras.optimizers import Adam

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),  # Update this line
              metrics=['accuracy'])


# In[19]:


#optimizer = Adam(lr = 0.001)


# In[20]:


optimizer = Adam(learning_rate=0.001)


# In[21]:


from tensorflow.keras.optimizers import Adam


# In[22]:


from keras.optimizers import Adam as adam_v2


# In[23]:


from tensorflow.keras.optimizers import Adam

# Define your learning rate
learning_rate = 0.001

# Create the optimizer
optimizer = Adam(learning_rate=learning_rate)

# Compile your model
model.compile(loss="binary_crossentropy", optimizer=optimizer)


# In[24]:


from tensorflow.keras.optimizers import Adam

# Compile your model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),  # Use learning_rate here
              metrics=['accuracy'])


# In[25]:


input_shape = (28, 28, 1)  # Height, Width, Channels


# In[26]:


input_shape = (10,)  # Just one dimension for features


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define your input shape
input_shape = (28, 28, 1)  # Example for grayscale images
# Or for tabular data
# input_shape = (10,)  # Example for 10 features

# Define your model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=input_shape))  # First layer
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])


# In[28]:


'''model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

nb_train_samples = 28709
nb_validation_samples = 3534
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)'''


# In[29]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define your input shape (example for images)
input_shape = (28, 28, 1)  # Adjust based on your data
num_classes = 10  # Example number of classes

# Define your model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=input_shape))  # Input layer
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the model with the correct argument
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),  # Use learning_rate here
              metrics=['accuracy'])

# Example for number of samples (adjust as needed)
nb_train_samples = 28709
nb_validation_samples = 3534


# # Driver Code

# In[30]:


from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


# In[31]:


import os
print(os.path.exists("E:\\Major Project\\EmotionDetectionModel.h5.keras"))


# In[32]:


face_classifier=cv2.CascadeClassifier("E:\\Major Project\\Validation_main\\haarcascade_frontalface.xml")
classifier = load_model("E:\\Major Project\\Validation_main\emotiondetector.h5")


# In[33]:


class_labels=['Angry','Happy','Neutral','Sad','Surprise']
cap=cv2.VideoCapture(0)


# In[34]:


"""while True:
    ret,frame=cap.read()"""


# In[35]:


import cv2

# Path to the Haar Cascade XML file
cascade_path = "E:\\Major Project\\Validation_main\\haarcascade_frontalface.xml"

# Load the cascade
face_classifier = cv2.CascadeClassifier(cascade_path)

# Check if the cascade is loaded successfully
if face_classifier.empty():
    print("Error loading cascade classifier")
else:
    print("Cascade classifier loaded successfully")


# In[58]:


frame = cv2.imread("E:\\Major Project\\Validation_main\\validation\\image_name.jpg")


# In[ ]:


while True:
    ret,frame=cap.read()


# In[ ]:


gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=face_classifier.detectMultiScale(gray,1.3,5)


# In[ ]:




