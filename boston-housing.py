#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 01:36:23 2021

@author: busrayildiz
"""

#Logistik olmayan regresyon yapıyoruz

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = np.loadtxt('housing.csv')
dataset_x = dataset[:, :-1]   #son sütun hariç tüm satırlar
dataset_y = dataset[:, -1]    #tüm satırlar, son sütun


training_dataset_x, test_dataset_x, training_dataset_y, test_dataset_y = train_test_split(dataset_x, dataset_y, test_size=0.20)

mms=MinMaxScaler()
mms.fit(training_dataset_x)
training_dataset_x = mms.transform(training_dataset_x)
test_dataset_x = mms.transform(test_dataset_x)   #y'ler üzerinde herhangi bir transform işlemi yapılmıyor


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential(name='BostonHousingPrices')
model.add(Dense(100, input_dim=training_dataset_x.shape[1], activation='relu', name='Hidden-1'))
model.add(Dense(100, activation='relu', name='Hidden-2'))
model.add(Dense(1, activation='linear', name='Output'))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
hist = model.fit(training_dataset_x, training_dataset_y, batch_size=32, epochs=100 , validation_split=0.2)

import matplotlib.pyplot as plt

figure = plt.gcf()
figure.set_size_inches((15,5))
plt.title('Loss - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1,len(hist.history['loss'])+1), hist.history['loss'])
plt.plot(range(1,len(hist.history['val_loss'])+1), hist.history['val_loss'])
plt.legend(['Loss' ,'Validation Loss'])
plt.show()

figure = plt.gcf()
figure.set_size_inches((15,5))
plt.title('Mean Absolute Error - Epoch Graphics')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.plot(range(1,len(hist.history['mae'])+1), hist.history['mae'])
plt.plot(range(1,len(hist.history['val_mae'])+1), hist.history['val_mae'])
plt.legend(['Mean Absolute Error' ,'Validation Mean Absolute Error'])
plt.show()


test_result = model.evaluate(test_dataset_x, test_dataset_y)
for i in range(len(test_result)):
    print(f'{model.metrics_names[i]} ---> {test_result[i]} ')










