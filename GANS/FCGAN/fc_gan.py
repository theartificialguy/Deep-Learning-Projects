import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np 

np.random.seed(1)

path = "generated_images/"

(x_train, y_train), (x_test, y_test) = mnist.load_data()

noise_img_dim = 100
batch_size = 16
iterations = 3750
img_height = 28
img_width = 28
channels = 1
epochs = 20

optimizer = Adam(0.0002, 0.5)

x_train = x_train.reshape(len(x_train), img_height*img_width*1)

# Generating GAN model
def create_generator():
	model = Sequential()
	model.add(Dense(units=256, input_dim=noise_img_dim))
	model.add(Activation("elu"))
	model.add(Dense(units=512))
	model.add(Activation("elu"))
	model.add(Dense(units=1024))
	model.add(Activation("elu"))
	model.add(Dense(units=img_height*img_width*channels))
	model.add(Activation("tanh"))
	model.compile(optimizer=optimizer, loss="binary_crossentropy")
	return model

def create_discriminator():
	model = Sequential()
	model.add(Dense(units=1024, input_dim=img_height*img_width*channels))
	model.add(Activation("elu"))
	model.add(Dense(units=512))
	model.add(Activation("elu"))
	model.add(Dense(units=256))
	model.add(Activation("elu"))
	model.add(Dense(units=1))
	model.add(Activation("sigmoid"))
	model.compile(optimizer=optimizer, loss="binary_crossentropy")
	return model

generator = create_generator()
discriminator = create_discriminator()

discriminator.trainable = False

gan_input = Input(shape=(noise_img_dim,))
fake_img = generator(gan_input)
gan_output = discriminator(fake_img)

GAN = Model(inputs=gan_input, outputs=gan_output)
GAN.compile(optimizer=optimizer, loss="binary_crossentropy")

# Display functions
def show_images(noise, epoch=None):
    generated_images = generator.predict(noise)
    plt.figure(figsize=(10, 10))
    for i, image in enumerate(generated_images):
        plt.subplot(10, 10, i+1)
        if channels == 1:
            plt.imshow(image.reshape((img_height, img_width)), cmap='gray')
        else:
            plt.imshow(image.reshape((img_height, img_width, channels)))
        plt.axis('off')
    plt.tight_layout()
    if epoch != None:
        plt.savefig(f'{path}/gan-images_epoch-{epoch}.png')

# Training process
const_noise = np.random.normal(0, 1, size=(100, noise_img_dim))

print("[INFO] Training...")
for i in range(epochs):
	for batch in range(iterations):
		noise = np.random.normal(0, 1, size=(batch_size, noise_img_dim))
		fake_img = generator.predict(noise)
		real_img = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
		x = np.concatenate((real_img, fake_img))
		disc_y = np.zeros(2*batch_size)
		disc_y[:batch_size] = 0.9
		dLoss = discriminator.train_on_batch(x, disc_y)
		y_gen = np.ones(batch_size)
		gLoss = GAN.train_on_batch(noise, y_gen)
	print("Epoch: {0}/{1}, Discriminator Loss: {2}, Generator Loss: {3}".format(i, epochs, dLoss, gLoss))
	if i%5 == 0:
		show_images(const_noise, i)

discriminator.save('fc_discriminator.h5')
generator.save('fc_generator.h5')
