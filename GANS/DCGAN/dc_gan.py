import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, Dropout
from tensorflow.keras.layers import Conv2DTranspose, Activation, Flatten
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np 

np.random.seed(10)

path = "generated_images/"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

noise_img_dim = 100
batch_size = 16
iterations = 3750
img_height = 32
img_width = 32
channels = 3
epochs = 20

optimizer = Adam(0.0002, 0.5)

x_train = x_train[np.where(y_train == 3)[0]]
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(len(x_train), img_height, img_width, channels)

# Generating GAN model
def create_generator():
	size = 4
	model = Sequential()
	model.add(Dense(units=size*size*256,kernel_initializer=RandomNormal(0, 0.02),input_dim=noise_img_dim))
	model.add(LeakyReLU(0.2))
	model.add(Reshape(target_shape=(size, size, 256)))
	model.add(Conv2DTranspose(128, (4,4), strides=2, padding="same", kernel_initializer=RandomNormal(0, 0.02)))
	model.add(LeakyReLU(0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=2, padding="same", kernel_initializer=RandomNormal(0, 0.02)))
	model.add(LeakyReLU(0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=2, padding="same", kernel_initializer=RandomNormal(0, 0.02)))
	model.add(LeakyReLU(0.2))
	model.add(Conv2D(channels, (3,3), padding="same", kernel_initializer=RandomNormal(0, 0.02)))
	model.add(Activation("tanh"))
	model.compile(optimizer=optimizer, loss="binary_crossentropy")
	return model

def create_discriminator():
	model = Sequential()
	model.add(Conv2D(64, (3,3), strides=2, padding="same", kernel_initializer=RandomNormal(0,0.02),
	 						input_shape=(img_height, img_width, channels)))
	model.add(LeakyReLU(0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(128, (3,3), strides=2, padding="same", kernel_initializer=RandomNormal(0,0.02)))
	model.add(LeakyReLU(0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(128, (3,3), strides=2, padding="same", kernel_initializer=RandomNormal(0,0.02)))
	model.add(LeakyReLU(0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Conv2D(256, (3,3), strides=2, padding="same", kernel_initializer=RandomNormal(0,0.02)))
	model.add(LeakyReLU(0.2))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(units=1, input_shape=(img_height, img_width, channels)))
	model.add(Activation("sigmoid"))
	model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
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
		gen_img = generator.predict(noise)
		sample = np.random.randint(0, x_train.shape[0], size=batch_size)
		real_img = x_train[sample]
		combined = np.concatenate((real_img, gen_img))
		disc_y = np.zeros(2*batch_size)
		disc_y[:batch_size] = 0.9
		dLoss = discriminator.train_on_batch(combined, disc_y)
		y_gen = np.ones(batch_size)
		gLoss = GAN.train_on_batch(noise, y_gen)
	print("Epoch: {0}/{1}, Discriminator Loss: {2}, Generator Loss: {3}".format(i, epochs, dLoss, gLoss))
	if i%10 == 0:
		show_images(const_noise, i)

discriminator.save('dc_discriminator.h5')
generator.save('dc_generator.h5')
