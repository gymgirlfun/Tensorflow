import os
import numpy as np
import glob
import shutil

import tensorflow as tf

import matplotlib.pyplot as plt

_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

# Move diff class into diff sub dirs
# 80% training data and 20% validation
for cl in classes:
  img_path = os.path.join(base_dir, cl)
  images = glob.glob(img_path + '/*.jpg')
  print("{}: {} Images".format(cl, len(images)))
  train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      os.makedirs(os.path.join(base_dir, 'train', cl))
    shutil.move(t, os.path.join(base_dir, 'train', cl))

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    shutil.move(v, os.path.join(base_dir, 'val', cl))
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Data Augmentation on training dataset
batch_size = 100
IMG_SHAPE = 150
image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
  rotation_range=45,zoom_range=0.5, horizontal_flip=True, height_shift_range=0.15, width_shift_range=0.15, rescale=1./255)
train_data_gen = image_gen_train.flow_from_directory(
  train_dir, batch_size=batch_size, target_size=(IMG_SHAPE, IMG_SHAPE), shuffle=True, class_mode="sparse")

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

# Create a generator for validation dataset, no need to augmentation, no need to shuffle
image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(val_dir, batch_size=batch_size, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode="sparse")

# Create CNN model
model = tf.keras.models.Sequential([
                                 tf.keras.layers.Conv2D(16, (3,3), padding='same', activation="relu", input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
                                 tf.keras.layers.MaxPooling2D((2,2)),
                                 tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu"),
                                 tf.keras.layers.MaxPooling2D((2,2)),
                                 tf.keras.layers.Conv2D(64, (3,3), padding='same', activation="relu"),
                                 tf.keras.layers.MaxPooling2D((2,2)),

                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dropout(0.2),
                                 tf.keras.layers.Dense(512, activation="relu"),
                                 tf.keras.layers.Dense(5, activation="softmax")])

# Compile the model
model.compile(optimizer='ADAM', metrics=['accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
print(model.summary())

# Train the model
epochs = 80
history = model.fit_generator(train_data_gen, 
                              epochs=epochs,
                              steps_per_epoch=int(np.ceil(train_data_gen.n / float(batch_size))),
                              validation_data=val_data_gen
                              validation_steps=int(np.ceil(val_data_gen.n / float(batch_size))))


# Plot metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print (acc)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
# print(type(epochs))
# print(type(epochs_range))

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(epochs_range, acc, label="Training Set Accuracy")
ax1.plot(epochs_range, val_acc, label="Validation Set Accuracy")
ax1.set_title("Accuracy")
ax1.legend(loc="lower right")

ax2.plot(epochs_range, loss, label="Training Set Loss")
ax2.plot(epochs_range, val_loss, label="Validation Set Loss")
ax2.set_title("Loss")
ax2.legend(loc = "upper left")
    
    
