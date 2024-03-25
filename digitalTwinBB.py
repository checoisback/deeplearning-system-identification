# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:46:31 2024

@author: bio
"""

from matplotlib import pyplot as plt
import scipy.io as sio
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Bidirectional, Input, Flatten, Activation, Reshape, RepeatVector, Concatenate, BatchNormalization, Masking
from keras.callbacks import ModelCheckpoint,EarlyStopping




X = pd.read_csv(os.path.join('data','baseline_fastabs_sioff','input.txt')).transpose()
Y = pd.read_csv(os.path.join('data','baseline_fastabs_sioff','output.txt')).transpose()


scalerX = StandardScaler()
scalerY = StandardScaler()

Xstd = scalerX.fit_transform(X)
Ystd = scalerY.fit_transform(Y)

Xtrain = Xstd[0:80,:]
Ytrain = Ystd[0:80,:]

Xval = Xstd[80:90,:]
Yval = Ystd[80:90,:]

Xtest = Xstd[90:,:]
Ytest = Ystd[90:,:]

# Build feedforward neural network model
model = Sequential()
model.add(Dense(units=8, input_shape=(Xtrain.shape[1],)))  # Output layer with 1 neuron for regression task
model.add(Dense(units=4))  # Output layer with 1 neuron for regression task
model.add(Dense(units=2))  # Output layer with 1 neuron for regression task
model.add(Dropout(0.2))
model.add(Dense(units=143))  # Output layer with 1 neuron for regression task

model.compile(optimizer='adam', loss='mae')
model.summary()
history = model.fit(Xtrain, Ytrain, validation_data= (Xval,Yval), epochs=1000, verbose=1)

plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')

yhat = model.predict(Xtrain)

plt.figure()
plt.plot(Ytrain[:,1])
plt.plot(yhat[:,1])