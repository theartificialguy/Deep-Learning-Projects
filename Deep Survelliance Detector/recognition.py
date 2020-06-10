from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = load_model("/home/yash/Desktop/PyImageSearch/checkpoints/emotion1.h5")
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img_size = 48

validation_dir = "/home/yash/Desktop/PyImageSearch/deep_survelliance_detector/fer2013/validation/"

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(validation_dir,
								color_mode="grayscale",
								target_size=(img_size,img_size),
								batch_size=32,
								class_mode="categorical",
								shuffle=False)

#getting class labels
class_labels = validation_generator.class_indices
class_labels = {v:k for v,k in class_labels.items()}

#predicting
test_img = "/home/yash/Downloads/happy1.jpg"

def get_label(prediction):
	for key, val in class_labels.items():
		if prediction == val:
			return key
	return -1

def predict(test_img):
	img = cv2.imread(test_img,cv2.IMREAD_GRAYSCALE)
	faces = classifier.detectMultiScale(img,scaleFactor=1.2,minNeighbors=7)
	face = []
	for (x,y,w,h) in faces:
		roi_gray = img[y:y+h,x:x+w]
		roi = cv2.resize(roi_gray, (img_size,img_size), interpolation=cv2.INTER_AREA)
		face.append(roi)
	num_image = np.array(face, dtype=np.float32)
	num_image /= 255.0
	num_image = num_image.reshape(1,48,48,1)
	predicted = model.predict(num_image)[0] #returns a list of probabilities of diff classes
	pred = predicted.argmax() #getting the max value in the list
	label = get_label(pred)
	return label

pred_class = predict(test_img)
original_image = mpimg.imread(test_img)
plt.xlabel("Predicted: {0}".format(str(pred_class)))
plt.imshow(original_image)
plt.show()
