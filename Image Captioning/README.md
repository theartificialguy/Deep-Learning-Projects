# Image Captioning
Image captioning is describing an image fed to the model. The task of object detection has been studied for a long time but recently the task of image captioning is coming into light.

## Requirements
1. Tensorflow
2. Keras
3. Numpy
4. h5py
5. Pandas
6. Pillow

## Output
The output of the model is a caption to the image

## Results
Following are a few results obtained after training the model for 70 epochs.

Image | Caption 
--- | --- 
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/beach.jpg" width="400"> | **Generated Caption:** A brown dog is running in the water.
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/image.jpg" width="400"> | **Generated Caption:** A tennis player hitting the ball.
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/child.png" width="400"> | **Generated Caption:** A child in a helmet is riding a bike.
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/street.png" width="400"> | **Generated Caption:** A group of people are walking on a busy street.

In some cases the classifier got confused and on blurring an image it produced bizzare results

Image | Caption
--- | ---
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/img1.jpg" width="400"> | **Generated Caption:** A brown dog and a brown dog are playing with a ball in the snow.
<img src="https://github.com/Shobhit20/Image-Captioning/blob/master/test/img1_blur.jpg" width="400"> | **Generated Caption:** A little girl in a white shirt is running on the grass.

#### Saved Model
https://drive.google.com/file/d/173TSx6uPHa7TlVDarifn0m2NUek0jWUv/view?usp=sharing
