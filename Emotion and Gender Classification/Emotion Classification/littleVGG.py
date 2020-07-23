import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Activation, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

#Data Augmentation
train_dir = "/home/yash/Desktop/PyImageSearch/deep_survelliance_detector/fer2013/train/"
validation_dir = "/home/yash/Desktop/PyImageSearch/deep_survelliance_detector/fer2013/validation/"

img_size = 48

train_datagen = ImageDataGenerator(rescale=1./255,
								   rotation_range=30,
								   shear_range=0.3,
								   zoom_range=0.3,
								   width_shift_range=0.4,
								   height_shift_range=0.4,
								   horizontal_flip=True,
								   fill_mode="nearest")

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
								color_mode="grayscale",
								target_size=(img_size,img_size),
								batch_size=32,
								class_mode="categorical",
								shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_dir,
								color_mode="grayscale",
								target_size=(img_size,img_size),
								batch_size=32,
								class_mode="categorical",
								shuffle=True)

num_classes = 6

#littleVGG or VGG9
#------------------------------------------------------------------
model = Sequential()
#1st layer
model.add(Conv2D(64,kernel_size=(3,3),padding="same",input_shape=(img_size,img_size,1)))
model.add(Activation("elu"))
model.add(BatchNormalization())
#2nd layer
model.add(Conv2D(64,kernel_size=(3,3)))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
#3rd layer
model.add(Conv2D(128,kernel_size=(3,3),padding="same"))
model.add(Activation("elu"))
model.add(BatchNormalization())
#4th layer
model.add(Conv2D(128,kernel_size=(3,3)))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
#5th layer
model.add(Conv2D(256,kernel_size=(3,3),padding="same"))
model.add(Activation("elu"))
model.add(BatchNormalization())
#6th layer
model.add(Conv2D(256,kernel_size=(3,3)))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))

model.add(Flatten())
#7th layer
model.add(Dense(256))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#8th layer
model.add(Dense(256))
model.add(Activation("elu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
#9th layer
model.add(Dense(num_classes))
model.add(Activation("softmax"))
#-------------------------------

#Creating callbacks
earlyStop = EarlyStopping(monitor='val_loss',
						  min_delta=0,
						  patience=3,
						  verbose=1,
						  restore_best_weights=True)

checkpoint = ModelCheckpoint('/home/yash/Desktop/PyImageSearch/checkpoints/emotion1.h5',
							 monitor='val_loss',
							 mode='min',
							 save_best_only=True,
							 verbose=1)

reduceLR = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks = [earlyStop, checkpoint, reduceLR]

model.compile(loss="categorical_crossentropy",
			metrics=["accuracy"],
			optimizer=Adam(lr=0.001))

nb_train_samples = 28273
nb_validation_samples = 3534
epochs = 25

history = model.fit_generator(
		train_generator,
		steps_per_epoch=nb_train_samples//32,
		epochs=epochs,
		callbacks=callbacks,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples//32)

#classification report
Y_pred = model.predict_generator(validation_generator,nb_validation_samples//32+1)
y_pred = np.argmax(Y_pred,axis=1)

class_labels = validation_generator.class_indices
class_labels = {v:k for v,k in class_labels.items()}
targets = list(class_labels.values())
print(classification_report(validation_generator.classes,y_pred,target_names=targets))

#visualizing losses and accuracy...
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

#plotting loss and accuracy...
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,25),train_loss,label="Train_Loss")
plt.plot(np.arange(0,25),val_loss,label="Val_Loss")
plt.plot(np.arange(0,25),train_acc,label="Train_Acc")
plt.plot(np.arange(0,25),val_acc,label="Val_Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
