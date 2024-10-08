# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:26:17 2024

@author: Francesco
"""


from models import lstm_model
from numpy import cumsum, zeros, random, float32
from matplotlib import pyplot
import scipy.io as sio
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# PARAMETERS
# =============================================================================
# Data generation parameters
num_samples = 100 #1024
num_timesteps = 144 #64
split_ratio = 0.1

# LSTM model parameters
model_shape= [64,64]
num_lookback = 1
num_epochs = 100
# =============================================================================



# LOAD DATA FROM VIRTUAL SUBJECT
# =============================================================================
print('Loading data...')
scalerY = MinMaxScaler()
y_dataRaw = pd.read_csv(os.path.join('data','baseline_fastabs_sioff','glucose.txt'))
y_dataRaw = y_dataRaw.to_numpy()
#y_dataRaw = y_dataRaw - y_dataRaw[0,:]
y_dataScaled = scalerY.fit_transform(y_dataRaw)
#y_dataScaled = y_dataScaled - y_dataScaled[0,:]
y_dataScaled = y_dataScaled.transpose()
y_data = y_dataScaled.reshape(num_samples, num_timesteps, 1)

scalerX1 = MinMaxScaler()
scalerX2 = MinMaxScaler()
scalerX3 = MinMaxScaler()
x1_dataRaw = pd.read_csv(os.path.join('data','baseline_fastabs_sioff','basal.txt'))
x1_dataRaw = x1_dataRaw.to_numpy()
x1_dataScaled = x1_dataRaw.transpose() #scalerX1.fit_transform(x1_dataRaw).transpose()
x1_data = x1_dataScaled.reshape(num_samples,num_timesteps,1)

x2_dataRaw = pd.read_csv(os.path.join('data','baseline_fastabs_sioff','bolus.txt'))
x2_dataRaw = x2_dataRaw.to_numpy()
x2_dataScaled = x2_dataRaw.transpose() #scalerX2.fit_transform(x2_dataRaw).transpose()
x2_data = x2_dataScaled.reshape(num_samples,num_timesteps,1)

x3_dataRaw = pd.read_csv(os.path.join('data','baseline_fastabs_sioff','carbs.txt'))
x3_dataRaw = x3_dataRaw.to_numpy()
x3_dataScaled = x3_dataRaw.transpose() #scalerX3.fit_transform(x3_dataRaw).transpose()
x3_data = x3_dataScaled.reshape(num_samples,num_timesteps,1)

x_data = np.concatenate((x2_data,x3_data), axis = 2)

# Split training and test data
x_test = x_data[:int(num_samples*split_ratio):,]
y_test = y_data[:int(num_samples*split_ratio):,]
x_train = x_data[int(num_samples*split_ratio):,]
y_train = y_data[int(num_samples*split_ratio):,]

# x_test = x_data[-1:,]
# y_test = y_data[-1:,]
# x_train = x_data[0:,]
# y_train = y_data[0:,]


print('x_train.shape..:', x_train.shape)
print('y_train.shape..:', y_train.shape)


# Plot one output signal
pyplot.figure()
pyplot.subplot(3,1,1)
pyplot.plot(y_train[0,], 'b')
pyplot.plot(y_train[1,], 'g')

pyplot.xlabel('Time')
pyplot.ylabel('Glucose [mg/dL]')
pyplot.grid()

# Plot one input signal
pyplot.subplot(3,1,2)
pyplot.plot(x_train[0,:,0], 'b')
pyplot.plot(x_train[1,:,0], 'g')

pyplot.xlabel('Time')
pyplot.ylabel('insulin')
pyplot.legend(loc='best')
pyplot.grid()

# # Plot one input signal
pyplot.subplot(3,1,3)
pyplot.plot(x_train[0,:,1], 'b')
pyplot.plot(x_train[1,:,1], 'g')

pyplot.xlabel('Time')
pyplot.ylabel('carbs')
pyplot.legend(loc='best')
pyplot.grid()

# # Plot one input signal
# pyplot.subplot(4,1,4)
# pyplot.plot(x_train[0,:,2], 'b')
# pyplot.plot(x_train[1,:,2], 'g')

# pyplot.xlabel('Time')
# pyplot.ylabel('carbs')
# pyplot.legend(loc='best')
# pyplot.grid()


pyplot.tight_layout()
pyplot.show()
# =============================================================================



# TRAIN LSTM MODEL WITH GENERATED MODEL
# =============================================================================
print('Training lstm model...')

num_u = x_train.shape[2]
num_y = y_train.shape[2]

# Creates LSTM model
lstm = lstm_model(model_shape, num_lookback, num_u, num_y)
lstm.fit(x_train, y_train, num_epochs)
# =============================================================================
# lstm_model = lstm.fitSimulation(x_train, y_train, 10, num_lookback)


#%% TEST LSTM MODEL AND PLOT
# =============================================================================
print('Plotting results...')

# LSTM model predicts mass position
y_pred = zeros(y_test.shape)
# ypred = lstm.predictLSTM(x_test,y_test)

# pyplot.figure()
# pyplot.plot(np.arange(0,288),y_test[0,], 'b', label='target')
# pyplot.plot(np.arange(num_lookback,288+num_lookback),ypred[0:288,0], 'r', label='LSTM')

for sample_index in range(x_test.shape[0]):
    for time_index in range(num_lookback,x_test.shape[1]): #x_test.shape[1]
        if time_index == num_lookback:
            x0 = y_train[sample_index,:time_index,0]
            u0 = x_train[sample_index,:time_index,:]
            y_pred[sample_index, time_index] = lstm.update_v1(x0,u0)
        else:
            y_pred[sample_index, time_index] = lstm.update(x_test[sample_index,time_index,:])


figure, (ax1, ax2, ax3) = pyplot.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
ax1.plot(y_test[0,], 'b', label='target')
ax1.plot(y_pred[0,], 'r', label='LSTM')
ax1.set_xlabel('Time')
ax1.set_ylabel('Glucose [mg/dL]')
ax1.legend(loc='best')
ax1.grid()

ax2.plot(x_test[0,:,0])
ax2.set_ylabel('Bolus')
ax2.grid()

ax3.plot(x_test[0,:,1])
ax3.set_ylabel('Carbs')
ax3.grid()
pyplot.tight_layout()
pyplot.show()