from keras.models import load_model
import numpy as np
import cv2
#import matplotlib.pyplot as plt

img_size = 28

classes = ["0","1","2","3","4","5","6","7","8","9"]

path = '/home/yash/Desktop/sample_image.png'
img = cv2.imread(path)
#print(img.shape)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
resized = cv2.resize(gray,(img_size,img_size))
image = np.array(resized).astype("float32")
image/=255.0
image = image.reshape(-1,28,28,1)

model = load_model("shallowNET_Digits.h5")

predicted = model.predict_classes(image)
print(predicted)

image1 = cv2.resize(img,(480,480))
cv2.putText(image1,"Label: {0}".format(str(classes[int(predicted)])),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,
	(0,255,0))

cv2.imshow("output",image1)
cv2.waitKey(0)
cv2.destroyAllWindows()
