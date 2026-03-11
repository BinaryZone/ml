import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras_preprocessing.image import ImageDataGenerator

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# read data
data_train_path = '../../data/defect-detection/train'
data_train_path = pathlib.Path(data_train_path)

data_test_path = '../../data/defect-detection/val'
data_test_path = pathlib.Path(data_test_path)

# label
CLASS_NAMES = np.array(['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc'])

# set photos size and batch size
BATCH_SIZE = 64
IMG_HEIGHT = 32
IM_WIDTH = 32

# normalization data
image_generator = ImageDataGenerator(rescale=1./255)
# generate image data
train_data_gen = image_generator.flow_from_directory(directory=str(data_train_path),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IM_WIDTH),
                                                     classes=list(CLASS_NAMES))

test_data_gen = image_generator.flow_from_directory(directory=str(data_test_path),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IM_WIDTH),
                                                     classes=list(CLASS_NAMES))

# build model
model = keras.Sequential()
model.add(Conv2D(filters=6, kernel_size=5, input_shape=(IMG_HEIGHT, IM_WIDTH, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=16, kernel_size=5, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(filters=120, kernel_size=5, activation='relu'))
model.add(Flatten())
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=6, activation='softmax'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
history = model.fit(train_data_gen, validation_data=test_data_gen, epochs=50)

# save model
model.save('convolutional_neural_network.h5')

# draw loss and accuracy photo
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('cnn model loss')
plt.legend()
plt.show()

# draw loss and accuracy photo
plt.plot(history.history['accuracy'], label='train_loss')
plt.plot(history.history['val_accuracy'], label='val_loss')
plt.title('cnn model loss')
plt.legend()
plt.show()