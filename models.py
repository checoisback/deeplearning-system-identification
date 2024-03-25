#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 22:36:20 2019

@author: eadali
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Bidirectional, Input, Flatten, Activation, Reshape, RepeatVector, Concatenate, BatchNormalization, Masking
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import load_model

import numpy  as np
from scipy.integrate import odeint
from numpy import sin, copy, zeros, float32

from matplotlib import pyplot




class pendulum_model:
    def __init__(self, m, l, b, g, x_0):
        """Inits pendulum constants and initial state

        # Arguments
            m: Pendulum mass
            l: Pendulum length
            b: Pendulum friction coeff
            g: Earth gravity acceleration
            x_0: Pendulum initial state
        """

        self.m = m
        self.l = l
        self.b = b
        self.g = g

        self.x_0 = x_0



    def ode(self, x, t, u):
        """Dynamic equations of pendulum

        # Arguments
            x: [angle of pendulum, angular velocity of pendulum]
            t: Time steps for ode solving
            u: External force applied to the pendulum

        # Returns
            Derivative of internal states
        """

        # Calculates equation coeffs
        c_1 = -self.b/(self.m*self.l**2)
        c_2 = -self.g/self.l
        c_3 = 1.0/(self.m*self.l**2)

        # ODE of pendulum
        theta, omega = x
        dxdt = [omega, c_1*omega + c_2*sin(theta) + c_3*u]

        return dxdt



    def update(self, u):
        """Interface function for pendulum model

        # Arguments
            u: External force applied to the pendulum

        # Returns
            Angle of pendulum
        """

        # Solving ODE with scipy library
        x = odeint(self.ode, self.x_0, [0,0.1], args=(u,))

        self.x_0 = x[1]

        return x[1,0]





class mass_spring_damper_model:
    def __init__(self, m, k, b, x_0):
        """Inits pendulum constants and initial state

        # Arguments
            m: Mass
            k: Spring coeff
            b: Friction coeff
            x_0: Initial state
        """

        self.m = m
        self.k = k
        self.b = b
        self.x_0 = x_0



    def ode(self, x, t, u):
        """Dynamic equations of pendulum

        # Arguments
            x: [position of mass, velocity of mass]
            t: Time steps for ode solving
            u: External force applied to the mass

        # Returns
            Derivative of internal states
        """

        # ODE of mass-spring-damper model
        pos, acc = x
        dxdt = [acc, -(self.b/self.m)*acc - (self.k/self.m)*pos + (1/self.m)*u]

        return dxdt



    def update(self, u):
        """Interface function for pendulum model

        # Arguments
            u: External force applied to the mass

        # Returns
            Position of mass
        """

        # Solving ODE with scipy library
        x = odeint(self.ode, self.x_0, [0,0.1], args=(u,))

        self.x_0 = x[1]

        return x[1,0]





class lstm_model:
    def __init__(self, model_shape, num_lookback, num_u, num_y):
        """Inits lstm model parameters

        # Arguments
            model_shape: List of cell number for each layer
            num_lookback: Number of lookback
            num_u: Number of inputs
            num_y: Number of predictions
        """
        # Input features of LSTM model
        self.x = zeros((1,num_lookback,num_u+num_y))

        # Creates LSTM model
        num_x = num_u + num_y
        num_layers = len(model_shape)

        self.model = Sequential()

        if self._equal(num_layers, 1):
            num_cells = model_shape[0]
                
            # self.model.add(Conv1D(filters=32, kernel_size=5, activation='elu',input_shape=(num_lookback,num_x)))
            # self.model.add(Conv1D(filters=16, kernel_size=3, activation='elu'))
            
            
            # #self.model.add(BatchNormalization())
            # self.model.add(MaxPooling1D()) # default = 2
            # self.model.add(Flatten())
            # self.model.add(RepeatVector(num_cells))
            self.model.add(LSTM(num_cells, activation='elu', return_sequences=True))
            self.model.add(LSTM(num_cells))
        else:
            num_cells = model_shape[0]
            self.model.add(LSTM(num_cells, activation='elu',input_shape=(num_lookback,num_x),
                                return_sequences=True))

            for num_cells in model_shape[1:-1]:
                
                self.model.add(LSTM(num_cells, activation='elu', return_sequences=True))

            num_cells = model_shape[-1]
            self.model.add(LSTM(num_cells))

        self.model.add(Dense(num_y))
        self.model.compile(loss='mse', optimizer='adam')

        # self.model.summary()



    def _equal(self, val_1, val_2):
        """Equality check function

        # Arguments
            val_1: First value for equality
            val_2: Second value for equality

        # Returns
            Equality result
        """

        condition_1 = (val_1 > (val_2-0.0001))
        condition_2 = (val_1 < (val_2+0.0001))

        return condition_1 and condition_2



    def _reshape(self, x_data, y_data):
        """Reshapes training data for LSTM

        # Arguments
            x_data: Features data
            y_data: Prediction data

        # Returns
            Reshaped x_data and y_data
        """

        x_data = copy(x_data)
        y_data = copy(y_data)

        # Gets dimension sizes from LSTM model
        _, num_lookback, num_x = self.model.layers[0].input_shape
        _, num_y = self.model.layers[-1].output_shape

        # Creates a new x_data
        new_shape = (x_data.shape[0], x_data.shape[1]-num_lookback,
                     num_lookback, num_x)
        x_data_new = zeros(new_shape, dtype=float32)

        x_data = x_data[:,1:,]

        # Fills new x_data
        for time_index in range(x_data_new.shape[1]):
            x_data_new[:,time_index,:,0:num_x-num_y] = x_data[:,time_index:time_index+num_lookback]
            #x_data_new[:,time_index,:,num_x-num_y:num_x] = y_data[:,time_index:time_index+num_lookback]#+num_lookback
            y_masked = y_data[:, time_index:time_index + num_lookback]
            mask = np.zeros_like(y_masked)
            mask[:, -2:] = 1  # Mask the last 2 timesteps
            x_data_new[:, time_index, :, num_x - num_y:num_x] = y_masked * mask

        # Creates a new y data
        y_data_new = y_data[:,num_lookback:,]

        return x_data_new, y_data_new
    

    def fit(self, x_data, y_data, num_epochs, validation_split=0.2):
        """Trains LSTM model

        # Arguments
            x_data: Features data
            y_data: Prediction data
            num_epochs: Number of epochs
            validation_split: Number of validation sample / Number of training sample
        """

        x_data = copy(x_data)
        y_data = copy(y_data)

        # Reshapes data for LSTM model
        x_data, y_data = self._reshape(x_data, y_data)

        _, num_lookback, num_x = self.model.layers[0].input_shape
        _, num_y = self.model.layers[-1].output_shape

        x_data = x_data.reshape(-1, num_lookback, num_x)
        y_data = y_data.reshape(-1)


        # Trains LSTM model
        checkpoint = ModelCheckpoint('temp_model.h5', save_best_only=True)
        checkpoint2 = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        self.history = self.model.fit(x_data, y_data, epochs=num_epochs,
                       verbose=1, validation_split=validation_split, batch_size = 288,
                       callbacks=[checkpoint,checkpoint2])
        
        pyplot.figure()
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='validation')
        

        self.lstm_model = load_model('temp_model.h5')
        


    def update(self, u):
        """Interface function for LSTM model

        # Arguments
            u: Input value

        # Returns
            Prediction of LSTM model
        """
        # Fills input
        self.x[0,:-1,0:2] = self.x[0,1:,0:2]
        self.x[0,-1,0:2] = u

        # Predicts output
        y_pred = self.model.predict(self.x, verbose=0)

        # Fills output
        self.x[0,:-1,2] = self.x[0,1:,2]
        self.x[0,-1,2] = y_pred

        return y_pred[0]
    
    def update_v1(self, x0, u0):
        """Interface function for LSTM model

        # Arguments
            u: Input value

        # Returns
            Prediction of LSTM model
        """
        # Fills input
        self.x[0,:,0:2] = u0
        self.x[0,:,2] = x0

        # Predicts output
        y_pred = self.model.predict(self.x)

        # Fills output
        self.x[0,:-1,2] = self.x[0,1:,2]
        self.x[0,-1,2] = y_pred

        return y_pred[0]
    

    def fitSimulation(self, x_data, y_data, num_epochs_sim, num_lookback):
        """Trains LSTM model for simulation

        # Arguments
            x_data: Features data
            y_data: Prediction data
            num_epochs: Number of epochs
            validation_split: Number of validation sample / Number of training sample
        """
        
        lstm = lstm_model([8, 4], num_lookback, 2, 1)
        lstm.fit(x_data, y_data, 5)


        # Define optimizer
        optimizer = tf.keras.optimizers.Adam()
        

        # Training loop
        for epoch in range(num_epochs_sim):
            
            y_pred = zeros(y_data.shape)
            epoch_loss = 0.0
            
            with tf.GradientTape() as tape:
                for sample_index in range(x_data.shape[0]):
                    for time_index in range(num_lookback, x_data.shape[1]):                    
                        #Forward pass: generate simulations
                        if time_index == num_lookback:
                            x0 = y_data[sample_index,:time_index,0]
                            u0 = x_data[sample_index,:time_index,:]
                            y_pred[sample_index, time_index] = lstm.update_v1(x0,u0)  
                        else:
                            y_pred[sample_index, time_index] = lstm.update(x_data[sample_index,time_index,:])
                        
                # Compute loss using MSE
                print('here:', y_pred)
                loss = tf.keras.losses.mean_squared_error(y_data, y_pred)
                epoch_loss += loss
                # Compute gradients
                gradients = tape.gradient(loss, self.model.trainable_variables)
                # Update model parameters
                #optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            # Print epoch loss
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(x_data)}')    
            
        return lstm_model
    


if __name__ == '__main__':
    """Test of models
    """

    # Test of pendulum_model class
    pendulum = pendulum_model(m=1, l=1, b=0.25, g=9.8, x_0=[1,0])
    theta = list()

    for t in range(512):
        theta.append(pendulum.update(8))

    pyplot.plot(theta, label='theta(t)')
    pyplot.legend(loc='best')
    pyplot.xlabel('t')
    pyplot.grid()
    pyplot.show()

    # Test of mass_spring_damper_model class
    msd = mass_spring_damper_model(m=1, k=8, b=0.8, x_0=[1,0])
    pos = list()

    for t in range(512):
        pos.append(msd.update(0.4))

    pyplot.plot(pos, label='pos(t)')
    pyplot.legend(loc='best')
    pyplot.xlabel('t')
    pyplot.grid()
    pyplot.show()


