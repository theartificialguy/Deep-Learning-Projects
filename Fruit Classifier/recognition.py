import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

test_dir = '/home/yash/Desktop/PyImageSearch/Fruit_classifier/data/test-multiple_fruits/'
data_path = '/home/yash/Desktop/PyImageSearch/Fruit_classifier/data/train/'

model = load_model('fruit_classifier1.h5')

categories = []
for category in os.listdir(data_path):
	categories.append(str(category))

def preProcess(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (64,64))
	numpyImg = np.array(resized).astype('float32')
	numpyImg /= 255.0
	numpyImg = numpyImg.reshape(-1,64,64,1)
	return numpyImg

for img in os.listdir(test_dir):
	readImg = cv2.imread(os.path.join(test_dir,img))
	rgbImage = cv2.cvtColor(readImg, cv2.COLOR_BGR2RGB)
	image = preProcess(readImg)
	print(int(model.predict_classes(image)))
	plt.Text(10,20,"Label: {0}".format(str(categories[int(model.predict_classes(image))])))
	plt.imshow(rgbImage)
	
	# cv2.putText(readImg, "Label: {0}".format(str(categories[int(model.predict_classes(image))])), 
	# 			(10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
	# cv2.imshow("output", readImg)
	# cv2.waitKey(0)
	plt.show()