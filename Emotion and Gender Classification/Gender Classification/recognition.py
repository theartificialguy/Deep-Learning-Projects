import tensorflow as tf 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np 

gender = load_model('gender_vgg16.h5')

classes = ['men', 'women']

imgPath1 = 'men1.jpeg'
imgPath2 = 'women1.jpg'

img1 = load_img(imgPath2, color_mode='rgb', target_size=(224, 224))
img1 = img_to_array(img1)
img1 /= 255
img1 = np.array(img1, dtype='float32')
img1 = img1.reshape(1, 224, 224, 3)
preds_probs = gender.predict(img1)
pred = np.argmax(preds_probs, axis=1)[0]
conf = round(preds_probs[0][pred]*100, 2)

print("Predicted: {0}, with confidence: {1}%".format(classes[pred], str(conf)))