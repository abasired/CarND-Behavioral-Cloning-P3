#!/usr/bin/env python2

"""
Created on Sun Feb  4 16:41:33 2018

@author: ashishbasireddy
"""

import csv
import codecs
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

#-------------------------
# Reading lineas from CSV
#-------------------------
lines = []
with open('/home/workspace/CarND-Behavioral-Cloning-P3/behavourial_cloning/data/combined_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        #print(line)
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

# -----------------------------------------------------------------------
# Defining a genretor coroutine to improve the execution time of training
# Includes preprocssing batches of images.
# ------------------------------------------------------------------------
def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        #shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            images = []
            measuremants = []
            for line in batch_samples:
                for i in range(3):  
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/behavourial_cloning/data/IMG/' + filename 
                    image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
                    measuremant = float(line[3])
                    if (i == 0): 
                        images.append(image)
                        measuremants.append(measuremant)
                        if abs(measuremant) > 0.05:
                            images.append(cv2.flip(image,1))
                            measuremants.append(measuremant*-1.0)
                    if (i == 1) and abs(measuremant) > 0.3:
                        measuremant += 0.2
                        images.append(image)
                        measuremants.append(measuremant)
                        images.append(cv2.flip(image,1))
                        measuremants.append(measuremant*-1.0) 
                    if (i == 2) and abs(measuremant) > 0.3:
                        measuremant -= 0.2
                        images.append(image)
                        measuremants.append(measuremant)
                        images.append(cv2.flip(image,1))
                        measuremants.append(measuremant*-1.0)  
                        

            X_train = np.array(images)
            y_train = np.array(measuremants)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size=32)
        
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

#-------------------------------
# Model architecture
#-------------------------------

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((50,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#model.load_weights("./model_base.h5")

checkpoint = ModelCheckpoint(filepath="/home/workspace/CarND-Behavioral-Cloning-P3/behavourial_cloning/model_nvidea_1_4_dropout_left_right.h5", monitor='val_loss', save_best_only=True)

stopper = EarlyStopping(monitor='val_loss', min_delta=0.003, patience=3)
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=4, verbose=1, callbacks=[checkpoint, stopper])


#model.save('model_center_images.h5')

### print the keys contained in the history object
#print(history_object.history.keys())
#

