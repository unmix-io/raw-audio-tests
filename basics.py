# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import soundlib

X = np.array([
    [[1], [2], [3], [4], [5]],
    [[2], [3], [4], [5], [6]],
    [[3], [4], [5], [6], [7]],
    [[4], [5], [6], [7], [8]],
    [[5], [6], [7], [8], [9]]
])

y = np.array([[6],[7],[8],[9],[10]])

# define the LSTM model
#model = Sequential()
#model.add(SimpleRNN(1, return_sequences=False, input_shape=(5,1)))


model = Sequential()
model.add(LSTM(512, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='linear'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

#model.add(SimpleRNN(32))
#model.add(Dropout(0.2))
#model.add(Dense(1, activation='softmax'))
#model.compile(loss='mean_squared_error', optimizer='rmsprop')
# define the checkpoint
filepath="weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model

model.fit(X, y, epochs=1000, batch_size=5, callbacks=callbacks_list)

# save model to single file
model.save('lstm_model.h5')


