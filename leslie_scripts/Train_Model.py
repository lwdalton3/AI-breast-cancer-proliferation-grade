"""
Module description goes here. Some notes I found in the original script:

note for whatever reason efficient net hasn't been working-- empirically first
to try is ResNet101V2 then VGG19

"""

import os
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print("Failed to enable GPU tree growth for GPU 0 or 1st GPU.")


def calc_class_weights(train_generator):
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    return {class_id: max_val/num_images for class_id, num_images in
            counter.items()}


def get_model(model_name, input_size, output_size):
    if 'EfficientNet' in model_name:
        model = eval('efn.' + model_name)(weights='imagenet',
                                          include_top=False,
                                          input_shape=(input_size, input_size,
                                                       3))
    else:
        model = eval('tf.keras.applications.' + model_name)(
            weights='imagenet',
            include_top=False,
            input_shape=(input_size, input_size, 3))

    model.trainable = True

    if output_size == 1:
        activation_func = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation_func = 'softmax'
        loss = 'sparse_categorical_crossentropy'

    model = tensorflow.keras.Sequential([
        model,
        tensorflow.keras.layers.GlobalAveragePooling2D(),
        tensorflow.keras.layers.Dense(output_size,
                                      activation=activation_func)])
    model.compile(loss=loss, optimizer=SGD(lr=0.001, decay=0.00001),
                  metrics=['sparse_categorical_accuracy'])

    return model


def get_output_size(class_folders):
    if len(class_folders) <= 2:
        return 1, 'binary'
    else:
        return len(class_folders), 'sparse'


# Expects folder with three classes for high, low grade and stromal images
train_folder = 'TCGA_train_no_mayo'
image_size = 224
batch_size = 16

# Which pre-trained model to use (see https://keras.io/api/applications/)
model_architecture = 'ResNet101V2'
model_name = model_architecture + '_no_mayo.h5'
epochs = 30
patience = 10
test_size = 0.2

# Lets create the augmentation configuration
# This helps prevent overfitting and increase accuracy
train_datagen = ImageDataGenerator(rescale=1/255,
                                   #  rotation_range=10,
                                   #  width_shift_range=0.05,
                                   #  height_shift_range=0.05,
                                   #  shear_range=0.05,
                                   #  zoom_range=0.05,
                                   #  horizontal_flip=True,
                                   #  vertical_flip=True,
                                   fill_mode='nearest',
                                   validation_split=test_size)

# We do not augment validation data, we only perform normalization
test_datagen = ImageDataGenerator(rescale=1/255, validation_split=test_size)

class_folders = os.listdir(train_folder)
outputsize, class_mode = get_output_size(class_folders)

train_generator = train_datagen.flow_from_directory(directory=train_folder,
                                                    target_size=(image_size,
                                                                 image_size),
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode=class_mode,
                                                    shuffle=True, seed=42,
                                                    subset='training')

test_generator = test_datagen.flow_from_directory(directory=train_folder,
                                                  target_size=(image_size,
                                                               image_size),
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  class_mode=class_mode,
                                                  shuffle=True, seed=42,
                                                  subset='validation')

class_weights = calc_class_weights(train_generator)
print("Class weights:", class_weights)

train_steps_per_epoch = min(int(len(train_generator.classes)/batch_size),
                            int(1024/batch_size))
test_steps_per_epoch = min(int(len(test_generator.classes)/batch_size),
                           int(1024/batch_size))

print("Train steps per epoch", train_steps_per_epoch)
print("Test steps per epoch", test_steps_per_epoch)

model = get_model(model_architecture, image_size, outputsize)

early_stop = EarlyStopping(patience=patience)
checkpointer = ModelCheckpoint(monitor='val_loss', filepath=model_name,
                               verbose=1, save_best_only=True)
callbacks = [checkpointer, early_stop]

history = model.fit(train_generator, steps_per_epoch=train_steps_per_epoch,
                    epochs=epochs, validation_data=test_generator,
                    validation_steps=test_steps_per_epoch, callbacks=callbacks,
                    class_weight=class_weights)

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
