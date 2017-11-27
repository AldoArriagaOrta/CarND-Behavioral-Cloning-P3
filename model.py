import csv  # to read and store the lines of the driving log file
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import  Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# path
base_path='./Captured_data/'

lines = []

#Retrieval of data stored in comma separated values file
with open(base_path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Dataset split for validation images (20%)
train_samples, validation_samples = train_test_split(lines, test_size=0.2, shuffle=True)
correction = 0.3

#A generator was used to load the training images.
def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never ends
        sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            measurements = []

            for line in batch_lines:
                i = 0
                while (i < 3): #This loop helps loading left and right images
                    source_path = line[i].replace('\\', '/')
                    filename = source_path.split('/')[-1]
                    current_path = base_path+'IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_flipped = np.fliplr(image)
                    images.append(image)
                    images.append(image_flipped)
                    measurement = float(line[3])
                    if i == 1:
                        measurement = measurement + correction
                    if i == 2:
                        measurement = measurement - correction
                    measurement_flipped = -measurement
                    measurements.append(measurement)
                    measurements.append(measurement_flipped)
                    i += 1

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

#parameters for the train and validation data generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Convolutional model based on Nvidia's approach with additional dropout layers
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) #Image Normalization
model.add(Cropping2D(cropping=((55,20), (0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam') #ADAM optimizer is used, no need to tune learning rate
history_object=model.fit_generator(train_generator, samples_per_epoch= len(train_samples),verbose=2,
                    validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=100)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()