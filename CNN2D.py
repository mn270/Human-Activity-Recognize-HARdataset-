"""
2D CNN model using wavelet transform
"""
import pywt
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from tensorflow.python import keras
import tensorflow as tf
import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.layers.convolutional import AveragePooling2D
from tensorflow.python.keras.layers import SpatialDropout2D
from keras.callbacks import History
from sklearn import metrics
import random
from tensorflow.keras.callbacks import TensorBoard
from time import time

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
history = History()

activities_description = {
    1: 'walking',
    2: 'walking upstairs',
    3: 'walking downstairs',
    4: 'sitting',
    5: 'standing',
    6: 'laying'
}

def read_signals(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data

def read_labels(filename):
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(lambda x: int(x)-1, activities))
    return activities

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

INPUT_FOLDER_TRAIN = '/home/marcin//Pobrane/HARDataset/train/Inertial Signals/'
INPUT_FOLDER_TEST = '/home/marcin/Pobrane/HARDataset/test/Inertial Signals/'

INPUT_FILES_TRAIN = ['body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
                     'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
                     'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt']

INPUT_FILES_TEST = ['body_acc_x_test.txt', 'body_acc_y_test.txt', 'body_acc_z_test.txt',
                     'body_gyro_x_test.txt', 'body_gyro_y_test.txt', 'body_gyro_z_test.txt',
                     'total_acc_x_test.txt', 'total_acc_y_test.txt', 'total_acc_z_test.txt']

LABELFILE_TRAIN = '/home/marcin/Pobrane/HARDataset/train/y_train.txt'
LABELFILE_TEST = '/home/marcin/Pobrane/HARDataset/test/y_test.txt'

train_signals, test_signals = [], []

for input_file in INPUT_FILES_TRAIN:
    signal = read_signals(INPUT_FOLDER_TRAIN + input_file)
    train_signals.append(signal)
train_signals = np.transpose(np.array(train_signals), (1, 2, 0))

for input_file in INPUT_FILES_TEST:
    signal = read_signals(INPUT_FOLDER_TEST + input_file)
    test_signals.append(signal)
test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

train_labels = read_labels(LABELFILE_TRAIN)
test_labels = read_labels(LABELFILE_TEST)

[no_signals_train, no_steps_train, no_components_train] = np.shape(train_signals)
[no_signals_test, no_steps_test, no_components_test] = np.shape(test_signals)
no_labels = len(np.unique(train_labels[:]))

print("The train dataset contains {} signals, each one of length {} and {} components ".format(no_signals_train, no_steps_train, no_components_train))
print("The test dataset contains {} signals, each one of length {} and {} components ".format(no_signals_test, no_steps_test, no_components_test))
print("The train dataset contains {} labels, with the following distribution:\n {}".format(np.shape(train_labels)[0], Counter(train_labels[:])))
print("The test dataset contains {} labels, with the following distribution:\n {}".format(np.shape(test_labels)[0], Counter(test_labels[:])))

uci_har_signals_train, uci_har_labels_train = randomize(train_signals, np.array(train_labels))
uci_har_signals_test, uci_har_labels_test = randomize(test_signals, np.array(test_labels))

scales = range(1,65)
waveletname =  'mexh'         #'mexh' 'morl'
train_size = 7352
train_data_cwt = np.ndarray(shape=(train_size, 64, 64, 9),dtype = 'float32')

for ii in range(0,train_size):
    if ii % 1000 == 0:
        print(ii)
    for jj in range(0,9):
        signal = uci_har_signals_train[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:64]
        train_data_cwt[ii, :, :, jj] = coeff_




test_size = 2947
test_data_cwt = np.ndarray(shape=(test_size,64, 64, 9),dtype = 'float32')
for ii in range(0,test_size):
    if ii % 1000 == 0:
        print(ii)
    for jj in range(0,9):
        signal = uci_har_signals_test[ii, :, jj]
        coeff, freq = pywt.cwt(signal, scales, waveletname, 1)
        coeff_ = coeff[:,:64]
        test_data_cwt[ii, :, :, jj] = coeff_
x_train = train_data_cwt
y_train = list(uci_har_labels_train[:train_size])
x_test = test_data_cwt
y_test = list(uci_har_labels_test[:test_size])
img_x = 64
img_y = 64
img_z = 9
num_classes = 6

batch_size = 368
epochs = 16

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
input_shape = (img_x, img_y, img_z)

# convert the data to the right type
#x_train = x_train.reshape(x_train.shape[0], img_x, img_y, img_z)
#x_test = x_test.reshape(x_test.shape[0], img_x, img_y, img_z)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
ytitle =  {1: 'f (Oś X)',
    2: 'f (Oś Y)',
    3: 'f (Oś Z)'}
axtitle =  {1: 'Acceleration',
    2: 'Angular velocity',
    3: 'Acceleration with G',}
activities_description = {
    1: 'Walk',
    2: 'Walk up',
    3: 'Walk down',
    4: 'Sitting',
    5: 'Standing',
    6: 'Laying'
}
x = [81, 137, 120,193,162,212 ]
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def scale_data(x_train, x_test):
    tdata_transformed_train = np.zeros_like(x_train)
    tdata_transformed_test = np.zeros_like(x_test)



    for i in range(img_z):
        s = StandardScaler()
        flatTrainX  = x_train[:, :, :, i].reshape(train_size, img_x*img_y) # make it a bunch of row vectors
        flatTestX = x_test[:, :, :, i].reshape(test_size, img_x * img_y)
        long = np.zeros((train_size+test_size,img_x*img_y))
        long[0:train_size,:] = x_train[:, :, :, i].reshape(train_size, img_x*img_y)
        long[train_size : test_size+train_size] = x_test[:, :, :, i].reshape(test_size, img_x * img_y)
        s.fit_transform(flatTrainX)
        flatTrainX = s.transform(flatTrainX)
        flatTestX = s.transform(flatTestX)
        flatTrainX  = flatTrainX .reshape(train_size, img_x, img_y) # reshape it back to tiles
        flatTestX = flatTestX.reshape(test_size, img_x, img_y)  # reshape it back to tiles
        tdata_transformed_train[:, :, :, i] = flatTrainX # put it in the transformed array
        tdata_transformed_test[:, :, :, i] = flatTestX
    return tdata_transformed_train, tdata_transformed_test

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
testy_norm = y_test
y_test = keras.utils.to_categorical(y_test, num_classes)

activation = "relu"

model = Sequential()

#Najelpszy
model.add(Conv2D(16, 5, activation=activation, padding='same',input_shape=input_shape))
model.add(Conv2D(32, 5, activation=activation, padding='same',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, 5, activation=activation, padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, 5, activation=activation, padding='same'))
model.add(SpatialDropout2D(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

# 3 Full connected layer
model.add(Dense(128, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(64, activation=activation))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#tensorboard = TensorBoard(log_dir="logs_conv2d_meh/{}".format(time()), histogram_freq=1, write_images=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

repeat = 10
LOSS = []
ACC = []
for i in range(repeat):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.15)

    train_score = model.evaluate(x_train, y_train, verbose=0)
    print('Train loss: {}, Train accuracy: {}'.format(train_score[0], train_score[1]))
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss: {}, Test accuracy: {}'.format(test_score[0], test_score[1]))
    ACC.append(test_score[1])
    LOSS.append(test_score[0])

print(LOSS)
print(ACC)
print('Acc:',sum(ACC)/repeat,'+/-',(max(ACC)-min(LOSS))/2)
print('Loss:',sum(LOSS)/repeat,'+/-',(max(LOSS)-min(LOSS))/2)

predictions = model.predict_classes(x_test)
confusion_matrix = metrics.confusion_matrix(y_true=testy_norm, y_pred=predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print(metrics.classification_report(testy_norm, predictions))

width = 12
height = 12
# fig, ax = plt.subplots()
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalized to the entire test set [%])")
plt.colorbar()
tick_marks = np.arange(6)
LABELS = ["Walk", "Walk up", "Walk down", "Sitting", "Standing", "Laying"]
plt.ylabel('Real value')
plt.xlabel('Prediction value')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accurancy')
plt.ylabel('Accurancy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()


