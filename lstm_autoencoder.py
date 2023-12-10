#!/usr/bin/env Python

"""
title:
     LSTM autoencoder to recreate sequence of missing data.
description:
author: Akshay Kale
"""
__author__ = "Akshay Kale"
__copyright__ = "GPL"
__email__ = "akale@unomaha.edu"

import numpy
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv3D
#from tensorflow.keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.keras.layers import ConvLSTM2D
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.backend import expand_dims
from tensorflow.keras.backend import repeat_elements
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot

def prepare_dataset(n=1000):
    # Load dataset
    data = numpy.load("mnist_test_seq.npy")

    # swap frames and observations to [obs, frames, height, width, channels]
    data = data.swapaxes(0, 1)

    # select first 100 observation to reduce memory
    subsetData = data[:n, :, :, :]

    # add channel dimension (grayscale)
    subsetData = numpy.expand_dims(subsetData, 4)

    # normalize to 0, 1
    subsetData[subsetData<128] = 0
    subsetData[subsetData>=128] = 1
    return subsetData

def shift_data(data, n_frames=15):
    X = data[:, 0:n_frames, :, :, :]
    y = data[:, 1:n_frames, :, :, :]
    #y = data[:, n_frames:(n_frames+n_frames), :, :, :]
    #y = data[:, 1:n_frames :, :, :]
    return X, y

# Reconstruction LSTM autoencoder:
    # The following reconstruction lstm autoencoder learns to reconstruct each input sequence.

def reconstruction_lstm(image_height, image_width, sequence_length):
    # ConvLSTM2D: A recurrent layer specifically used for images that passed hidden state to next layer.
    # LSTM input is a 3D (samples, timesteps, features)
    # Convoultion input is a 4D (samples, channels, rows, cols)
    # ConvLSTM2D input is a 5D (samples, timesteps, channels, rows, cols) 
    # Batch normalization: standardizes the inputs to a layer for each mini-batch.
    # Used to accelerate the training, regularization, reducing generalization error.

    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 1),
                         input_shape=(None, image_height, image_width, 1),
                         padding='same', return_sequences=True,
                         activation='relu'))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(2, 2),
                         padding='same', return_sequences=True,
                         activation='relu'))
    model.add(BatchNormalization())
    #model.add(Lambda(lambda x: repeat_elements(expand_dims(x, axis=1), sequence_length, 1)))
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 1),
                         padding='same', return_sequences=True,
                         activation='relu'))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(2, 2),
                         padding='same', return_sequences=True,
                         activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                         activation='sigmoid',
                         padding='same', data_format='channels_last'))
    #model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    plot_model(model, show_shapes=True, to_file='reconstruction_lstm_autoencoder.png')
    return model

# Prediction LSTM autoencoder
    # We are modifying the reconstruction LSTM autoencoder to instead prediction the next step in the sequence
def prediction_lstm(image_height, image_width, sequence_length):

    # ConvLSTM2D: A recurrent layer specifically used for images that passed hidden state to next layer.
    # LSTM input is a 3D (samples, timesteps, features)
    # Convoultion input is a 4D (samples, channels, rows, cols)
    # ConvLSTM2D input is a 5D (samples, timesteps, channels, rows, cols) 

    # Batch normalization: standardizes the inputs to a layer for each mini-batch.
    # Used to accelerate the training, regularization, reducing generalization error.

    # Encoding
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 1),
                         input_shape=(None, image_height, image_width, 1),
                         padding='same', return_sequences=True,
                         activation='relu'))
    model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=64, kernel_size=(2, 2),
                         padding='same',
                         activation='relu'))
    model.add(BatchNormalization())

    model.add(Lambda(lambda x: repeat_elements(expand_dims(x, axis=1), sequence_length-1, 1)))

    # Decoding
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 1),
                         padding='same', return_sequences=True,
                         activation='relu'))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=64, kernel_size=(2, 2),
                         padding='same', return_sequences=True,
                         activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv3D(filters=1, kernel_size=(1, 1, 1),
                         activation='sigmoid',
                         padding='same', data_format='channels_last'))
    #model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])
    plot_model(model, show_shapes=True, to_file='prediction_lstm_autoencoder.png')
    return model

def composite_lstm_autoencoder(image_height, image_width, sequence_length):
    visible = Input(shape=(None, image_height, image_width, 1))
    # define encoder
    encoder = ConvLSTM2D(filters=64, kernel_size=(2, 2),
                         padding='same',
                         activation='relu')(visible)

    # define reconstruct decoder:
    decoder1 = Lambda(lambda x: repeat_elements(expand_dims(x, axis=1), sequence_length, 1))(encoder)
    # Lambda layer is similar to RepeatVector
    decoder1 = ConvLSTM2D(filters=64, kernel_size=(2, 2),
                          padding='same', return_sequences=True,
                          activation='relu')(decoder1)
    decoder1 = TimeDistributed(Dense(1))(decoder1)

    # define predict decoder
    decoder2 = Lambda(lambda x: repeat_elements(expand_dims(x, axis=1), sequence_length, 1))(encoder)
    decoder2 = ConvLSTM2D(filters=64, kernel_size=(2, 2),
                          padding='same', return_sequences=True,
                          activation='relu')(decoder2)
    decoder2 = TimeDistributed(Dense(1))(decoder2)
    model = Model(inputs=visible, outputs=[decoder1, decoder2])
    model.compile(optimizer='adam', loss='mse')
    plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')
    return model

def compare_results(groundTruth, predictions):
    #compare results
    for i in range(0, 15):
        fig =  pyplot.figure(figsize=(10, 5))
        ax = fig.add_subplot(122)
        ax.text(1, -2, ('Ground truth at time:' + str(i)), fontsize=15, color='black')
        toplot_true = groundTruth[0, i, ::, ::, 0]
        pyplot.imshow(toplot_true)

        # predictions
        ax = fig.add_subplot(121)
        ax.text(1, -2, ('Predicted frame at time:' + str(i)), fontsize=15, color='black')
        toplot_pred = predictions[0, i, ::, ::, 0]
        pyplot.imshow(toplot_pred)
        filename = 'compare_frame_%d.png' % (i)
        pyplot.savefig(filename)
    pyplot.close()

def compare_results_prediction(groundTruth, predictions):
    #compare results
    for i in range(0, 14):
        fig =  pyplot.figure(figsize=(10, 5))
        ax = fig.add_subplot(122)
        ax.text(1, -2, ('Ground truth at time:' + str(i)), fontsize=15, color='black')
        toplot_true = groundTruth[0, i, ::, ::, 0]
        pyplot.imshow(toplot_true)

        # predictions
        ax = fig.add_subplot(121)
        ax.text(1, -2, ('Predicted frame at time:' + str(i)), fontsize=15, color='black')
        toplot_pred = predictions[0, i, ::, ::, 0]
        pyplot.imshow(toplot_pred)
        filename = 'compare_frame_%d.png' % (i)
        pyplot.savefig(filename)
    pyplot.close()

def summarize_diagnostics(history):
      pyplot.figure(figsize=(20, 10))
      pyplot.subplot(211)
      pyplot.title('Class entropy loss')
      loss = pyplot.plot(history.history.history['loss'], color='blue', label='train')
      valLoss = pyplot.plot(history.history.history['val_loss'], color='orange', label='test')
      #pyplot.legend([loss, valLoss], ['train', 'test'])
      pyplot.legend()
      filename = 'loss_plot.png'

      # plot accuracy
      pyplot.subplot(212)
      pyplot.title('Classification Accuracy')
      acc = pyplot.plot(history.history.history['accuracy'], color='blue', label='train')
      valAcc = pyplot.plot(history.history.history['val_accuracy'], color='orange', label='test')
      filename = 'accuracy_plot.png'
      pyplot.legend()
      pyplot.savefig(filename)
      pyplot.close()


def run():
    data = prepare_dataset()
    sequence_length = 15
    image_height = data.shape[2]
    image_width = data.shape[3]
    print("\nImage height and width: ", image_height, image_width)

    # prepare dataset
    X, y = shift_data(data, sequence_length)

    #model = reconstruction_lstm(image_height, image_width, sequence_length)
    model = prediction_lstm(image_height, image_width, sequence_length)
    #model = composite_lstm_autoencoder(image_height, image_width, sequence_length)
    model.summary()
    print("Sequence length:%d, height:%d, width:%d" % (sequence_length, image_height, image_width))

    # fit model
    model.fit(X, y, batch_size=12, epochs=50, validation_split=0.05)
    #model.fit([X, X], [X, y], batch_size=32, epochs=10, verbose=0)

    # evaluate model
    loss, acc, mse = model.evaluate(X, y, verbose=1)
    print("Accuracy: %0.01f%%" % (acc*100))
    print("loss: %0.01f%%" % (loss*100))
    print("mse: %0.01f%%" % (mse*100))

    # training history
    summarize_diagnostics(model)

    # select a random obervation
    test_set = numpy.expand_dims(X[5, :, :, :, :], 0)

    # predict
    prediction = model.predict(test_set)

    # compare results
    compare_results_prediction(test_set, prediction)

run()
