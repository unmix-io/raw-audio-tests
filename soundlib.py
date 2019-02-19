import random
import numpy as np

def getSample(duration = 0.1):

    # random frequency between 100 and 200
    f = random.randint(100,200)

    # Create sound samples
    volume = 0.5     # range [0.0, 1.0]
    fs = 44100       # sampling rate, Hz, must be integer
    #duration = 0.05   # in seconds, may be float
    #f = 200.0        # sine frequency, Hz, may be float

    # generate samples, note conversion to float32 array
    samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

    f= random.randint(300,400)
    samples2 = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

    both = samples + samples2
    # Keras
    both = both + 1
    samples = samples + 1

    #X = np.array(
    #    [[x] for x in both]
    #)
    #
    #y = np.array([
    #    samples
    #])
    X = both
    y = samples

    return (X,y)

