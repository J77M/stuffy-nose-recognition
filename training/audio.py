import numpy as np
import wave
import pyaudio
try:
    import utils
except:
    import training.utils as utils



class Audio(object):
    '''object for audio data collection'''
    def __init__(self):
        self.a = pyaudio.PyAudio()
        self.stream = self.a.open(format=pyaudio.paInt16, channels=1, rate=utils.RATE, input=True,
                        frames_per_buffer=utils.CHUNK, output=True)


    def process_data(self, raw_data):
        '''unpack and process data as follow:
            bandpass -> fft -> remove over utils.top_val frequency -> scale -> reduce resolution'''
        waveData = wave.struct.unpack("%dh" % (utils.CHUNK), raw_data)
        indata = np.array(waveData)
        indata = utils.bandpass(utils.band[0], utils.band[1], indata, utils.RATE)
        freq, y = utils.fft(indata, utils.RATE)
        maxIndex = np.where(freq > utils.top_val)[0][0]
        y = y[:maxIndex]
        y = y / np.amax(y)  # features scaling - need improvement
        yp = utils.reduce_resolution(y, mn=utils.RESOLUTION)
        return yp


    def start_stream(self, _class, time):
        '''_class - list of objects - calling them on new data'''
        if type(_class) != []:
            _class = [_class]

        for i in range(0, int(utils.RATE / utils.CHUNK * time)):
            raw_data = self.stream.read(utils.CHUNK)
            for cls in _class:
                data = self.process_data(raw_data)
                cls(data)

        self.stream.stop_stream()
        self.stream.close()
        self.a.terminate()