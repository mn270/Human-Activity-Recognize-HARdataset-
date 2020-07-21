from numpy import mean
from numpy import std
from numpy import dstack
from numpy import fft
from numpy import correlate
import numpy as np
from scipy import signal
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import pydot
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from tensorflow.python import keras
from sklearn import metrics
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling1D
from tensorflow.python.keras.layers import SpatialDropout1D
from tensorflow.python.keras.layers.convolutional import AveragePooling1D
from keras.utils import to_categorical
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras import regularizers
from sklearn.metrics import confusion_matrix

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix='/home/marcin/Pobrane/'):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')

    y1 = trainy[np.where(trainy == 4)]
    y2 = trainy[np.where(trainy == 5)]
    y3 = trainy[np.where(trainy == 6)]
    x1 = trainX[np.where(trainy == 4), :, :]
    x2 = trainX[np.where(trainy == 5), :, :]
    x3 = trainX[np.where(trainy == 6), :, :]
    trainy_new = np.concatenate((y1, y2, y3))
    trainX_new = np.concatenate((x1[0], x2[0], x3[0]))

    print(trainX.shape, trainy.shape)

    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')

    y1_ = testy[np.where(testy == 4)]
    y2_ = testy[np.where(testy == 5)]
    y3_ = testy[np.where(testy == 6)]
    x1_ = testX[np.where(testy == 4), :, :]
    x2_ = testX[np.where(testy == 5), :, :]
    x3_ = testX[np.where(testy == 6), :, :]

    testy_new = np.concatenate((y1_, y2_, y3_))
    testX_new = np.concatenate((x1_[0], x2_[0], x3_[0]))

    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy_new = trainy_new - 1
    testy_new = testy_new - 1
    # one hot encode y
    trainy_new = to_categorical(trainy_new)
    testy_norm = testy_new
    testy_new = to_categorical(testy_new)
    #    trainX, testX = Magnitude(trainX,testX)
    #    trainX, testX = AutoCorallation(trainX, testX)
    #    trainX, testX = Power_Spectrum(trainX,testX)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX_new, trainy_new, testX_new, testy_new, testy_norm


def Magnitude(trainX, testX):
    """
        Calculates the magnitude of signal
        :param trainX: (array)
        :param testX: (array)
        :return:
                trainX: (array)
                testX: (array)
        """
    trainX_fft = []
    testX_fft = []

    # Signal parameters
    N = 128
    f_s = 50
    t_n = 2.56
    T = t_n / N
    shape_test = testX.shape[2]
    shape_train = trainX.shape[2]

    for j in range(9):
        for i in range(trainX.shape[0]):
            trainX_fft.append(2.0 / 128 * abs(np.fft.fft(trainX[i, :, j])))
        for i in range(testX.shape[0]):
            testX_fft.append(abs(np.fft.fft(testX[i, :, j])))

        trainX_spectrum = np.array(trainX_fft)
        testX_spectrum = np.array(testX_fft)
        trainX_spectrum_2 = np.zeros((trainX.shape[0], 128, shape_train + 1 + j))
        testX_spectrum_2 = np.zeros((testX.shape[0], 128, shape_test + 1 + j))

        for i in range(trainX.shape[0]):
            trainX_spectrum_2[i] = np.concatenate((trainX[i], np.reshape(trainX_spectrum[i], (128, 1))), axis=1)
        for i in range(testX.shape[0]):
            testX_spectrum_2[i] = np.concatenate((testX[i], np.reshape(testX_spectrum[i], (128, 1))), axis=1)

        trainX = None
        del trainX
        trainX = trainX_spectrum_2
        testX = None
        del testX
        testX = testX_spectrum_2
        trainX_spectrum_2 = None
        del trainX_spectrum_2
        testX_spectrum_2 = None
        del testX_spectrum_2

        trainX_spectrum_2 = trainX[:, :, 9:18]
        testX_spectrum_2 = testX[:, :, 9:18]
    #    return  trainX_spectrum_2, testX_spectrum_2
    return trainX, testX


def AutoCorallation(trainX, testX):
    """
        Calculates the autocorrelation
        :param trainX: (array)
        :param testX: (array)
        :return:
                trainX: (array)
                testX: (array)
        """
    trainX_corr = []
    testX_corr = []

    # Signal parameters
    N = 128
    f_s = 50
    t_n = 2.56
    T = t_n / N
    shape_test = testX.shape[2]
    shape_train = trainX.shape[2]

    for j in range(9):
        for i in range(trainX.shape[0]):
            autocorr_values = np.correlate(trainX[i, :, j], trainX[i, :, j], mode='full')
            trainX_corr.append(autocorr_values[len(autocorr_values) // 2:])
        for i in range(testX.shape[0]):
            autocorr_values = np.correlate(testX[i, :, j], testX[i, :, j], mode='full')
            testX_corr.append(autocorr_values[len(autocorr_values) // 2:])

        trainX_spectrum = np.array(trainX_corr)
        testX_spectrum = np.array(testX_corr)
        trainX_spectrum_2 = np.zeros((trainX.shape[0], 128, shape_train + 1 + j))
        testX_spectrum_2 = np.zeros((testX.shape[0], 128, shape_test + 1 + j))

        for i in range(trainX.shape[0]):
            trainX_spectrum_2[i] = np.concatenate((trainX[i], np.reshape(trainX_spectrum[i], (128, 1))), axis=1)
        for i in range(testX.shape[0]):
            testX_spectrum_2[i] = np.concatenate((testX[i], np.reshape(testX_spectrum[i], (128, 1))), axis=1)

        trainX = None
        del trainX
        trainX = trainX_spectrum_2
        testX = None
        del testX
        testX = testX_spectrum_2
        trainX_spectrum_2 = None
        del trainX_spectrum_2
        testX_spectrum_2 = None
        del testX_spectrum_2

    #        trainX_spectrum_2 = trainX[:,:,9:18]
    #        testX_spectrum_2 = testX[:, :, 9:18]
    #    return  trainX_spectrum_2, testX_spectrum_2
    return trainX, testX


def Power_Spectrum(trainX, testX):
    """
        Calculates the power spectrum
        :param trainX: (array)
        :param testX: (array)
        :return:
                trainX: (array)
                testX: (array)
        """
    trainX_fft = []
    testX_fft = []

    # Signal parameters
    N = 128
    f_s = 50
    t_n = 2.56
    T = t_n / N
    shape_test = testX.shape[2]
    shape_train = trainX.shape[2]

    for j in range(9):
        for i in range(trainX.shape[0]):
            f_values, psd_values = signal.welch(trainX[i, :, j], fs=f_s)
            trainX_fft.append(psd_values[0:64])
        for i in range(testX.shape[0]):
            f_values, psd_values = signal.welch(testX[i, :, j], fs=f_s)
            testX_fft.append(psd_values[0:64])

        trainX_spectrum = np.array(trainX_fft)
        testX_spectrum = np.array(testX_fft)

        shape = np.shape(trainX_spectrum)
        padded_array_1 = np.zeros((shape[0], 128))
        padded_array_1[:shape[0], :shape[1]] = trainX_spectrum
        shape = np.shape(testX_spectrum)
        padded_array_2 = np.zeros((shape[0], 128))
        padded_array_2[:shape[0], :shape[1]] = testX_spectrum

        trainX_spectrum_2 = np.zeros((trainX.shape[0], 128, shape_train + 1 + j))
        testX_spectrum_2 = np.zeros((testX.shape[0], 128, shape_test + 1 + j))

        for i in range(trainX.shape[0]):
            trainX_spectrum_2[i] = np.concatenate((trainX[i], np.reshape(padded_array_1[i], (128, 1))), axis=1)
        for i in range(testX.shape[0]):
            testX_spectrum_2[i] = np.concatenate((testX[i], np.reshape(padded_array_2[i], (128, 1))), axis=1)

        trainX = None
        del trainX
        trainX = trainX_spectrum_2
        testX = None
        del testX
        testX = testX_spectrum_2
        trainX_spectrum_2 = None
        del trainX_spectrum_2
        testX_spectrum_2 = None
        del testX_spectrum_2

        trainX_spectrum_2 = trainX[:, :, 9:18]
        testX_spectrum_2 = testX[:, :, 9:18]
    #    return  trainX_spectrum_2, testX_spectrum_2
    return trainX, testX


def scale_data(trainX, testX):
    """
        Scale data 2D
         :param trainX: (array)
        :param testX: (array)
        :return:
                trainX: (array)
                testX: (array)
        """
    # remove overlap
    cut = int(trainX.shape[1] / 2)
    longX = trainX[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
    flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
    # standardize
    s = RobustScaler()
    # fit on training data
    s.fit(longX)
    # print("MEAN:")
    # print(s.mean_)
    # print("------------------------------------------")
    # print("VAR:")
    # print(s.var_)
    # print("------------------------------------------")
    # print("STD:")
    # print(s.scale_)

    print(s.get_params(True))
    # apply to training and test data
    longX = s.transform(longX)
    flatTrainX = s.transform(flatTrainX)
    flatTestX = s.transform(flatTestX)
    # reshape
    flatTrainX = flatTrainX.reshape((trainX.shape))
    flatTestX = flatTestX.reshape((testX.shape))
    return flatTrainX, flatTestX


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, testy_norm):
    """
    Create, fit and evaluate a model
    :param trainX: (array)
    :param trainy: (array)
    :param testX: (array)
    :param testy: (array)
    :param testy_norm: (array)
    :return:
        accurancy (float)
        loss (float)
    """
    verbose, epochs, batch_size = 1, 60, 16  # 16
    trainX, testX = scale_data(trainX, testX)
    #    trainX, testX = Magnitude(trainX,testX)
    #    trainX, testX = AutoCorallation(trainX, testX)
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    print(testX.shape)
    print(testy.shape)
    model = Sequential()

    # Small structure
    model.add(Conv1D(32, 5, activation='relu', padding='same', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 5, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, 5, activation='relu', padding='same'))
    model.add(SpatialDropout1D(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.summary()
    plot_model(model, 'model_info.png', show_shapes=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.15,
                        shuffle=True)
    # evaluate model
    loss, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    export_model(model)
    predictions = model.predict_classes(testX)
    print(metrics.classification_report(testy_norm, predictions))
    confusion_matrix = metrics.confusion_matrix(y_true=testy_norm, y_pred=predictions)
    print(confusion_matrix)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
    print("")
    print("Confusion matrix (normalised to % of total test data):")
    print(normalised_confusion_matrix)

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
    tick_marks = np.arange(3)
    LABELS = ["Sitting", "Standing", "Laying"]
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
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
    return accuracy, loss


# summarize scores
def summarize_results(scores, losses):
    """
    Summarize values from all test after training
    :param scores: (list) scores [%]
    :param losses: (list) losses [%]
    :return: (float)
    """
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    m, s = mean(losses), std(losses)
    print('Losses: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=10):
    """
    Main function
    """
    # load data
    trainX, trainy, testX, testy, testy_norm = load_dataset()
    # repeat experiment
    scores = list()
    losses = list()
    for r in range(repeats):
        score, loss = evaluate_model(trainX, trainy, testX, testy, testy_norm)
        score = score * 100.0
        loss = loss * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
        losses.append(loss)
    # summarize results
    summarize_results(scores, losses)


def export_model(model):
    """
    Save model (use 3 version h5, wb and tflite)
    """
    # model.save('path_to_my_model1.h5')
    # new_model = keras.models.load_model('path_to_my_model1.h5')

    # Save tf.keras model in HDF5 format.
    keras_file = "keras_model.h5"
    tf.keras.models.save_model(model, keras_file)
    new_model = keras.models.load_model('keras_model.h5')
    # Convert to TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    #   tf.keras.models.save_model(model,'model',save_format='tf')
    tf.saved_model.save(model, "nowy")
    # tf.saved_model.save()
    print("graph saved!")


# run the experiment
run_experiment()
