from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mlxtend.data import loadlocal_mnist
from keras.models import Sequential
from keras.layers import Conv2D,Activation,MaxPooling2D
from keras.layers import Flatten,Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use("ggplot")

classes = ["0","1","2","3","4","5","6","7","8","9"]

X, y = loadlocal_mnist(
        images_path='/home/yash/Desktop/PyImageSearch/DNN_DigitRecognition/train-images.idx3-ubyte', 
        labels_path='/home/yash/Desktop/PyImageSearch/DNN_DigitRecognition/train-labels.idx1-ubyte')

X = X.astype("float32")
X = X.reshape(-1,28,28,1)
X/=255.0 #normalizing
y = np_utils.to_categorical(y,10) #onehot encode

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#shallow net

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),padding="same",input_shape=X_train[0].shape))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))

sgd = SGD(lr=0.01,momentum=0.9,decay=1e-6,nesterov=True)

model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=["accuracy"])
print("[INFO] Model Training...")

hist = model.fit(X_train,y_train,batch_size=32, epochs=6,validation_data=(X_test,y_test),verbose=1)
model.save("shallowNET_Digits.h5")

predictions = model.predict(X_test,batch_size=32)
print(classification_report(y_test.argmax(axis=1),predictions.argmax(axis=1),target_names=classes))

train_loss = hist.history["loss"]
train_acc = hist.history["accuracy"]
val_loss = hist.history["val_loss"]
val_acc = hist.history["val_accuracy"]

plt.figure()
plt.plot(np.arange(0,6),train_loss,label="Train_Loss")
plt.plot(np.arange(0,6),val_loss,label="Val_Loss")
plt.plot(np.arange(0,6),train_acc,label="Train_Acc")
plt.plot(np.arange(0,6),val_acc,label="Val_Acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
