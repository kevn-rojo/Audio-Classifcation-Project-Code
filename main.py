#!pip install comet_ml
#import comet_ml
#comet_ml.init()
from comet_ml import Experiment



import IPython.display as ipd
import numpy as np
import pandas as pd

import librosa
import librosa.display

import matplotlib.pyplot as plt
import os

from scipy.io import wavfile as wav
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from datetime import datetime 
from tensorflow.keras.utils import to_categorical


#this is part of the Comet library, which helps visuallize charts
experiment = Experiment(project_name = "audio recognition") 



# we take the csv file from the data set and create a dataframe
# we then read the class column and make list with only the unique
# this ensures we have all the classifications in one list
dataFrame = pd.read_csv('UrbanSound8K.csv')  # read the csv file
labels = list(dataFrame['class'].unique())  # looks at the class column and returns a list of unique audio classes

# Let's grab a single audio file from each audio class
files = dict()  # we are essentially intializing a new dictionary called files

for i in range(len(labels)):
    tmp = dataFrame[dataFrame['class'] == labels[i]][:3].reset_index()  # the [:3] is that it wants oneto grab 3 items 
    path = 'UrbanSound8K/audio/fold{}/{}'.format(tmp['fold'][1], tmp['slice_file_name'][1])
    files[labels[i]] = path

  
################################ plot some waveforms ################################
  
#plot some 
file_name = 'UrbanSound8K/audio/fold1/191431-9-0-66.wav'
librosa_audio, librosa_sample_rate = librosa.load(files_name)  # the load function returns and audio time series and a sampling rate 
#if a sampling rate is not specified in the load function it is by default 22050
librosa.display.waveplot(librosa_audio, sr=librosa_sample_rate)  # this actually plot it without it it the plots are empty


################################ Extract MFCCs ################################

# function that extracts mean functions
def extract_mfcc_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed


features = []
full_database_path = 'UrbanSound8K/audio'

# extract the mfccs for evrey file in the database 
for index, row in dataFrame.iterrows():
    file_name = os.path.join(os.path.abspath(full_database_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    class_label = row["class"]
    data = extract_mfcc_features(file_name)
    features.append([data, class_label])


print('Finished Extracting all MFCCS')
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])


################################ Defining our input and output data sets ################################
X = np.array(featuresdf.feature.tolist()) # this is an array of your input features
y = np.array(featuresdf.class_label.tolist()) # this is an array of the answers which could be used to compare at the output
#of the neural network 



# Assign claasifications a uniqur binary number  
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) # returns a binary matrix of the output array


#randomly splits the  input features and the outputs into two sets a training set and a test set 
#The test set ~20% will be used to test our neural network, and the othe 80% will used to train it 
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 69)



################################ Build your sequential Neural Network ################################
num_labels = yy.shape
tf.keras.backend.clear_session() # if you do not clear,the previous models will still be there 
# and you will continue to build on the models.the line above clears all the models.

def create_sequential_neural_network( ):
    model = Sequential()
    model.add( Dense(256, input_shape = (30,)) )# you have to define the input shape for the first layer 
    model.add( Activation('relu') )
    model.add( Dropout(0.5) )

    # second layer 
    model.add( Dense(256) ) 
    model.add( Activation('relu') )
    model.add(Dropout(0.5))

    # output layer
    model.add( Dense(num_labels) )  
    model.add( Activation('softmax') )

    # make the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

# uses the function from above to define the the actual layer 
neural_network = create_sequential_neural_network() 

# Display model architecture summary 
neural_network.summary()

################################ Train and Test the neural Network ################################

#Calculate pre-training accuracy 
pre_training_score = neural_network.evaluate(x_test, y_test, verbose=0)
pre_training_accuracy = pre_traing_score[1] *100
print("pre training accuracy: %.4f%%" % pre_training_accuracy)


#The fit function will train your neural network
neural_network.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), verbose=1)


# Evaluating the model on the training  set 
traing_score = neural_network.evaluate(x_train, y_train, verbose=0)
training_accuracy = traing_score [1] *100
print("post training accuracy: %.4f%%" % training_accuracy)


#evaluating the model in the testing set
test_score = neural_network.evaluate(x_test, y_test, verbose=0)
test_accuracy = test_score [1] *100
print("post training accuracy: %.4f%%" % test_accuracy)
