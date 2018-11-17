import pyaudio
import numpy as np
import wave
from time import time
import utils

#LABEL = 1 if recording normal voice, LABEL = 0 if recording mumble voice (stuffy nose)
# set before recording !!!
LABEL = 1
# LABEL = int(input(':'))

RECORD_SECONDS = 45

band = (250,6000) # for bandpass
top_val = 6000 # max frequency

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 2*RATE # every 2 seconds. for all recordings have to be same
RESOLUTION = 12 #  factor for reducing resolution

data = []


def record(stream):
    raw_data = stream.read(CHUNK)
    waveData = wave.struct.unpack("%dh"%(CHUNK), raw_data)
    indata = np.array(waveData)
    indata = utils.bandpass(band[0],band[1],indata,RATE)
    freq, y = utils.fft(indata, RATE)
    maxIndex = np.where(freq > top_val)[0][0]
    y = y[:maxIndex]
    y = y / np.amax(y) # features scaling - need improvement
    yp = utils.reduce_resolution(y, mn= RESOLUTION)
    data.append(yp)


if __name__=="__main__":

    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                  frames_per_buffer=CHUNK, output = True)

    print('recording started')
    for i in range(0, int(utils.RATE / utils.CHUNK * RECORD_SECONDS)):
        record(stream)

    stream.stop_stream()
    stream.close()
    p.terminate()
    tim = time()
    np.save('data/data_x-{}'.format(tim),[np.array(data), LABEL])