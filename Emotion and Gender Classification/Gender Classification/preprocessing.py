import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import vgg16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import os
import cv2

data_path = '/content/gdrive/My Drive/data/'

num_classes = 2

cats = ['men', 'women']

features = []
labels = []

for label, cat in tqdm(enumerate(cats)):
  path = os.path.join(data_path, cat)
  for imgPath in os.listdir(path):
    try:
      image = load_img(os.path.join(path, imgPath), color_mode='rgb', target_size=(224, 224))
      img = img_to_array(image)
      img /= 255
      features.append(img)
      labels.append(label)
    except Exception as e:
      print('[ERROR] Loading image!')

X = np.array(features, dtype='float32')
X = X.reshape(len(X), 224, 224, 3)
y = np.array(labels)
y = to_categorical(y, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# transfer learning
vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg.layers:
  layer.trainable = False

top = vgg.output
top = GlobalAveragePooling2D()(top)
top = Dense(units=256, activation='relu')(top)
top = BatchNormalization()(top)
top = Dropout(0.2)(top)
top = Dense(units=128, activation='relu')(top)
top = BatchNormalization()(top)
top = Dropout(0.2)(top)
top = Dense(units=num_classes, activation='softmax')(top)

model = Model(inputs=vgg.input, outputs=top)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('/content/gdrive/My Drive/gender_vgg16.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

reducelr = ReduceLROnPlateau(monitor='val_loss',
                             factor=0.2,
                             patience=3,
                             verbose=1,
                             min_delta=0.0001)

callbacks = [checkpoint, reducelr]

hist = model.fit(x_train, y_train, epochs=40, batch_size=32, verbose=1, callbacks=callbacks, validation_data=(x_test, y_test))
