from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import Activation, MaxPooling2D
from tensorflow.keras.layers import Conv2D, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

pickle_in1 = open("X.pickle","rb")
X = pickle.load(pickle_in1)
pickle_in2 = open("y.pickle","rb")
y = pickle.load(pickle_in2)

#print(X[0].shape) # (64,64,1)

data_path = '/home/yash/Desktop/Fruit_classifier/data/train/'

target = []
for category in os.listdir(data_path):
	target.append(str(category))

num_classes = 76

earlyStop = EarlyStopping(monitor='val_loss',
						  min_delta=0,
						  patience=3,
						  verbose=1,
						  restore_best_weights=True)

checkpoint = ModelCheckpoint('/home/yash/Desktop/PyImageSearch/checkpoints/fruit_classifier1.h5',
							 monitor='val_loss',
							 mode='min',
							 save_best_only=True,
							 verbose=1)

reduceLR = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks = [earlyStop, checkpoint, reduceLR]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15)

#CNN model
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding="same",input_shape=X[0].shape))
model.add(Activation("relu"))
model.add(Conv2D(32,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size=(3,3),padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

#sgd = SGD(lr=0.01,momentum=0.9,decay=1e-6,nesterov=True)
rms = RMSprop(lr = 0.001)

model.compile(optimizer=rms,loss="categorical_crossentropy",metrics=["accuracy"])

print("[INFO] Model Training...")

hist = model.fit(X_train,y_train,callbacks=callbacks,batch_size=32,
				 epochs=30, validation_data=(X_test,y_test),verbose=1)
#model.save("speedNET_CATDOG.h5")

#evaluating model...
predictions = model.predict(X_test,batch_size=32)
print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1),target_names=target))

#visualizing losses and accuracy...
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

#plotting loss and accuracy...
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,20),train_loss,label="Train_Loss")
plt.plot(np.arange(0,20),val_loss,label="Val_Loss")
plt.plot(np.arange(0,20),train_acc,label="Train_Acc")
plt.plot(np.arange(0,20),val_acc,label="Val_Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
