
import keras.backend as k
from keras.layers import Dense, Activation, LSTM, Dropout, Reshape, Input, Lambda, RepeatVector
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.initializers import glorot_uniform

from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *


n_a = 128 # number of activation units. 

X, Y, n_values, indices_values = load_music_utils() # preprocessing 

# n_values : no. of unique values in the dataset(78) (similar to the character vocabulary)


dense = Dense(n_values, activation='softmax')
reshape = Reshape((1, 78))
lstm = LSTM(n_a, return_state = True)


def model(Tx, n_a, n_values):

    
    # Define the input of your model with a shape 
    X = Input(shape=(Tx, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    outputs = []
    
    for t in range(Tx):
        
        x = Lambda(lambda x: X[:,t,:])(X)
        x = reshape(x)

        a, _, c = lstm(x, initial_state=[a, c])

        out = dense(a)

        outputs.append(out)
        
    model = Model([X, a0, c0], outputs)
        
    return model	



m = model(30 , n_a = 128, n_values = 78)

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


m1 = 60
a0 = np.zeros((m1, n_a))
c0 = np.zeros((m1, n_a))

m.fit([X, a0, c0], list(Y), epochs=100)


def generate(lstm, dense, n_values = 78, n_a = 64, Ty = 100):

    x0 = Input(shape=(1, n_values))
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    
    a = a0
    c = c0
    x = x0

    outputs = []

    for t in range(Ty):

        a, _, c = lstm(x, initial_state=[a, c])
        
        out = dense(a)

        outputs.append(out)
        
        #  Select the next value according to "out", and set "x" to be the one-hot representation of the
        #  selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #  the line of code you need to do this. 
        x = Lambda(one_hot)(out)

    inference_model = Model([x0, a0, c0], outputs)
    
    
    return inference_model


music = generate(lstm, dense, n_values = 78, n_a = 128, Ty = 200)


x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

pred = music.predict([x_initializer, a_initializer, c_initializer])

indices = np.argmax(pred, 2)

results = to_categorical(indices, num_classes=None)

results, indices = predict_and_sample(music, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))

out_stream = generate_music(music)




'''
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)
'''
