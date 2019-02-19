import soundlib
import pyaudio
import numpy as np
import butterworth_bandpass

def play(arr):
    p = pyaudio.PyAudio()
    fs = 44100

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

    # play. May repeat with different volume values (if done interactively)
    stream.write((arr).tobytes())

    stream.stop_stream()
    stream.close()

    p.terminate()

def _load_data(n_prev = 100):
    X, y = soundlib.getSample(1)
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


def display_spectrogram(y):
    import librosa.display
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(np.asarray(y))), ref=np.max)
    librosa.display.specshow(D, x_axis='time', y_axis='log', sr=44100)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')


X,y = _load_data(300)
y2 = y
print(y)
from keras.models import load_model
# load model from single file
model = load_model('lstm_model.h5')
# make predictions
y = model.predict(X, verbose=0)
y = [i[0] for i in y-1]

import smoothing
#y = smoothing.smooth(y, window_len=100)
#y = butterworth_bandpass.butter_bandpass_filter(y, 100, 500, 44100, order=5)

display_spectrogram(np.asarray(y, dtype=np.float32))
display_spectrogram(np.array([i[0] for i in y2-1]))


print(y)
print(y2-1)
play(np.tile(np.array([i[0] for i in X-1]),1))
play(np.tile(np.array([i[0] for i in y2-1]),1))
play(np.tile(np.asarray(y, dtype=np.float32),1))


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 10.0, 0.001)
s = y[:10000]

fig, ax = plt.subplots()
ax.plot(t, s)

s2 = [i[0] for i in X-1][:10000]
#ax.plot(t, s2)

s3 = [i[0] for i in y2-1][:10000]
ax.plot(t, s3)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

#fig.savefig("test.png")
plt.show()
