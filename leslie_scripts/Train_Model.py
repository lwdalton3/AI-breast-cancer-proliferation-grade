
# Developed by Dalton Pathology; Austin,TX,2020
# command line   python3 Train_Model.py
# This is the one to use


#!/usr/bin/env python
# coding: utf-8
def calc_class_weights(train_generator):
    counter = Counter(train_generator.classes)                          
    max_val = float(max(counter.values()))       
    return {class_id : max_val/num_images for class_id, num_images in counter.items()}                     

# note for whatever reason efficient net hasn't been working-- empirically first to try is ResNet101V2 then VGG19 
def getModel(modelName, inputSize, outputSize): #Function to create VGG19 Model
    if "EfficientNet" in modelName:
        model = eval("efn."+modelName)(weights = "imagenet", include_top=False, input_shape = (inputSize, inputSize, 3)) #Get model
    else:
        model = eval("tf.keras.applications."+modelName)(weights = "imagenet", include_top=False, input_shape = (inputSize, inputSize, 3)) #Get model
    
    model.trainable = True # Full Training

    if outputSize == 1:
        activationFunc = "sigmoid"
        loss = "binary_crossentropy"
    else:
        activationFunc = "softmax"
        loss = "sparse_categorical_crossentropy"
    
    model = tensorflow.keras.Sequential([
        model,
        tensorflow.keras.layers.GlobalAveragePooling2D(),
        tensorflow.keras.layers.Dense(outputSize, activation=activationFunc)])
    
    
    model.compile(loss =loss, optimizer = SGD(lr=0.001, decay=0.00001), metrics=["sparse_categorical_accuracy"])
    return model

def getOutputSize(classFolders):
    #Get Output size
    if len(classFolders) <= 2:
        return 1,"binary"
    else:
        return len(classFolders),"sparse"
    
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try: tf.config.experimental.set_memory_growth(physical_devices[0], True)
except: print("Failed to enable GPU tree growth for GPU 0 or 1st GPU.") # Invalid device or cannot modify virtual devices once initialized.
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Flatten, Dropout, Dense
import tensorflow.keras #Import Base Keras model
from tensorflow.keras.optimizers import SGD,Adam
import numpy as np #Linear Regression
import matplotlib.pyplot as plt
from collections import Counter
import efficientnet.tfkeras as efn 

trainFolder = 'TCGA_train_no_mayo' #expects folder with three classes for high, low grade and stromal images
imageSize = 224# change2 what image size in pixels to use
batch_size= 16
modelArchitecture = "ResNet101V2"  # change_TC name of pre-trained models see https://keras.io/api/applications/
modelName = modelArchitecture+"_no_mayo.h5" #change model name here
epochs = 30# change number epochs to use   30 is good starting point
patience = 10  #change how many epochs without improvement before ending
test_size = 0.2# change fraction of train/validation in training

# Lets create the augmentation configuration
# This helps prevent overfitting and increase accuracy
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    #rotation_range=10,
                                    #width_shift_range=0.05,
                                    #height_shift_range=0.05,
                                    #shear_range=0.05,
                                    #zoom_range=0.05,
                                    #horizontal_flip=True,
                                    #vertical_flip=True,
                                    fill_mode='nearest',
                                    validation_split=test_size
                                  )
test_datagen  = ImageDataGenerator(rescale = 1./255, validation_split=test_size)  # We do not augment validation data. we only perform normalization

#===Code Parameters===#


classFolders = os.listdir(trainFolder)
outputsize,class_mode = getOutputSize(classFolders)


train_generator = train_datagen.flow_from_directory(
    directory=trainFolder,
    target_size=(imageSize, imageSize),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode=class_mode,
    shuffle=True,
    seed=42,
    subset='training'
)

test_generator = test_datagen.flow_from_directory(
    directory=trainFolder,
    target_size=(imageSize, imageSize),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode=class_mode,
    shuffle=True,
    seed=42,
    subset='validation'
)

class_weights = calc_class_weights(train_generator)
print("class_weights:",class_weights)
train_steps_per_epoch = min(int(len(train_generator.classes)/batch_size),int(1024/batch_size))
test_steps_per_epoch = min(int(len(test_generator.classes)/batch_size),int(1024/batch_size))

print("train steps_per_epoch",train_steps_per_epoch)
print("test steps_per_epoch",test_steps_per_epoch)

model = getModel(modelArchitecture,imageSize,outputsize)

early_stop = EarlyStopping(patience=patience)
checkpointer = ModelCheckpoint(monitor="val_loss", filepath=modelName, verbose=1, save_best_only=True)
callbacks = [checkpointer,early_stop]

history = model.fit(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              epochs=epochs,
                              validation_data=test_generator,
                              validation_steps=test_steps_per_epoch,
                              callbacks=callbacks,
                              class_weight=class_weights
                             )


acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
