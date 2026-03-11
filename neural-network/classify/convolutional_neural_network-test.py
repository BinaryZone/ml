import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2

# label
CLASS_NAMES = np.array(['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc'])

# set photos size and batch size
IMG_HEIGHT = 32
IM_WIDTH = 32

# load model
model = load_model('convolutional_neural_network.h5')

# predict
src = cv2.imread('../../data/defect-detection/val/In/In_3.bmp')
src = cv2.resize(src, (IMG_HEIGHT, IM_WIDTH))
src = src.astype('int32')
src = src / 255

# expand data
test_img = tf.expand_dims(src, 0)
print(test_img.shape)

preds = model.predict(test_img)
score = preds[0]

print('模型预测的结果为{}, 概率为{}'.format(CLASS_NAMES[np.argmax(score)], np.max(score)))