import pyaudio
import numpy as np
import wave
import tensorflow as tf
from training import utils


band = (250,6000)# for bandpass
top_val = 6000

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2*RATE # every 2 seconds. for all recordings have to be same
RESOLUTION = 12 #  factor for reducing resolution

RECORD_SECONDS = 45

model = tf.keras.models.load_model('training/trained_models/2-conv_layers-1-dense_layers-64-conv_size-256-dense_size-10-kernel-1542471350.h5')
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])


def soundPlot(stream):
    '''new data -> filtering with bandpass -> fft -> scaling -> reducing resolution -> predict'''
    data = stream.read(CHUNK)
    waveData = wave.struct.unpack("%dh"%(CHUNK), data)
    npArrayData = np.array(waveData)
    data = utils.bandpass(band[0],band[1],npArrayData,RATE)
    freq, y = utils.fft(data, fs = RATE)
    maxIndex = np.where(freq > top_val)[0][0]
    max_val = np.amax(y)
    y = y[ :maxIndex]
    y = y / max_val
    yp = utils.reduce_resolution(y, mn = RESOLUTION)
    result = model.predict(yp.reshape(-1, 1000, 1))
    print(round(result[0][0], 1))

if __name__=="__main__":
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                  frames_per_buffer=CHUNK, output = True)

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        soundPlot(stream)

    stream.stop_stream()
    stream.close()
    p.terminate()