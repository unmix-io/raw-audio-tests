from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import metrics
import numpy as np

training_length = 10000
rnn_size = 512
hm_epochs = 30

def generate_sequence(length=10):
    step = np.random.randint(0,50)
    first_element = np.random.randint(0,10)
    first_element = 0
    l_ist = [(first_element + (step*i)) for i in range(length)]
    return l_ist

training_set = []

for _ in range(training_length):
    training_set.append(generate_sequence(10))

feature_set = [i[:-1] for i in training_set]

label_set = [i[-1:] for i in training_set]

X = np.reshape(feature_set,(training_length, 9, 1))
y = np.array(label_set)

print(X)
print(y)

model = Sequential()
model.add(LSTM(rnn_size, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(rnn_size))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='linear'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

filepath="checkpoint_folder/weights-improvement.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X,y,epochs=hm_epochs, callbacks=callbacks_list)