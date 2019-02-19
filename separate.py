import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import soundlib


def _load_data(n_prev = 100):
    X, y = soundlib.getSample(0.1)
    y = np.array([[x] for x in y])
    data = X
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append([x for x in data[i:i+n_prev]])
        docY.append(y[i+n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

X = []
y = []

for i in range(1,10000):
    X_, y_ = _load_data(500)
    X.extend(X_)
    y.extend(y_)

X = np.asarray(X)
y = np.asarray(y)

print(X.shape)

# define the model
model = Sequential()
model.add(Dense(768, input_dim=500, init="uniform", activation="relu"))
model.add(Dense(768, activation="relu", kernel_initializer="uniform"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
# define the checkpoint
filepath="weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model

#for iteration in range(1,100):
#    print("iteration " + str(iteration))
#    X, y = _load_data(200)
#    print(y)
model.fit(X, y, epochs=1, batch_size=500, callbacks=callbacks_list)

# save model to single file
model.save('lstm_model.h5')
